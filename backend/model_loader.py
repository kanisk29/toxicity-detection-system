import torch
import json
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import time 
from groq import Groq
import os
from dotenv import load_dotenv

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL_PATH = "kanisk29/toxicity-detector-v1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


threshold_path = hf_hub_download(
    repo_id=MODEL_PATH,
    filename="thresholds.json"
)

with open(threshold_path, "r") as f:
    thresholds = json.load(f)

labels = list(thresholds.keys())
label_to_idx = {
    label: i
    for i, label in enumerate(labels)
}

# --------------------------- SHAP setup ---------------------------
def _shap_predict(texts):
    """
    Prediction function SHAP calls repeatedly with masked/perturbed
    versions of the input text. Must accept a list/array of strings
    and return a 2D array of shape (n_texts, n_labels).
    """
    inputs = tokenizer(
        list(texts),
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()
    return probs


# Text masker splits on the model's own tokenizer so perturbations
# stay aligned with how the model actually sees the text.
_masker = shap.maskers.Text(tokenizer)
explainer = None

def get_explainer():
    global explainer

    if explainer is None:
        explainer = shap.Explainer(
            _shap_predict,
            _masker,
            output_names=labels
        )

    return explainer

def explain_text(text, label_filter=None, top_k=8):
    """
    Returns the top_k most influential tokens for each requested label.

    label_filter: list of label names to explain (e.g. only the ones
                  that got flagged). Defaults to all labels.

    shap_value > 0  -> token pushed the prediction TOWARD toxic
    shap_value < 0  -> token pushed the prediction AWAY from toxic
    """
    shap_values = get_explainer()([text])

    tokens = shap_values.data[0]

    if isinstance(tokens, str):
        tokens = tokens.split()

    target_labels = label_filter if label_filter else labels
    explanations = {}

    for label in target_labels:
        idx = label_to_idx[label]
        values = shap_values.values[0]

        if len(values.shape) == 2:
            values = values[:, idx]
        ranked = sorted(
            zip(tokens, values),
            key=lambda pair: abs(pair[1]),
            reverse=True
        )[:top_k]

        explanations[label] = [
            {"token": str(tok).strip(), "shap_value": float(val)}
            for tok, val in ranked
            if str(tok).strip()
        ]

    return explanations


def explain_text_html(text):
    """
    Optional: full SHAP text plot as a standalone HTML string
    (handy for a debug endpoint or saving to a file — not needed
    for the JSON API response).
    """
    shap_values = get_explainer()([text])
    return shap.plots.text(shap_values, display=False)
# --------------------------------------------------------------------


def groq_llm(rewrite_content):
    prompt = f"""You are an expert rewriter of toxic comments into non toxic ones.
    Current toxic comment: {rewrite_content}
    Rules:
    - Do NOT use ANY OBSCENIITIES
    - ONLY CONVERT THE GIVEN TOXIC COMMENT INTO A NON TOXIC ONE AND DO NOT RETURN ANYTHING ELSE
    - Convey the message in a more pleasant tone 
    - Return only 1 suggestion on how to write the comment
    - Keep it short and concise under 50 words.
    Return: 
    Rewritten Comment Suggestion: <rewritten_comment>
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an expert rewriter of toxic comments into non toxic ones."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def predict_text(text, with_explanation=False):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.sigmoid(logits).cpu().numpy()[0]

    results = {}
    flagged = False
    flagged_labels = []

    for i, label in enumerate(labels):
        prob = float(probs[i])
        thresh = thresholds[label]
        pred = 1 if prob >= thresh else 0

        results[label] = {
            "confidence": prob,
            "prediction": pred
        }

        if prob >= thresh:
            flagged = True
            flagged_labels.append(label)

    response = {
        "predictions": results
    }

    if flagged:
        response["rewrite"] = groq_llm(text)

        if with_explanation:
            try:
                response["explanations"] = explain_text(
                    text,
                    label_filter=flagged_labels
                )
            except Exception as e:
                response["explanations"] = {
                    "error": str(e)
                }
    return response

