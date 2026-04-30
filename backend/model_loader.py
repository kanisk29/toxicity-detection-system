import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
from core.logging import get_logger
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

logger = get_logger(__name__)
logger.info("App started")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL_PATH = "kanisk29/toxicity-detector-v1"    

logger.info("Loading Tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
logger.info("Loaded Tokenizer")
logger.info("Loading Model")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
logger.info("Loaded Model")


threshold_path = hf_hub_download(
    repo_id=MODEL_PATH,
    filename="thresholds.json"
)

with open(threshold_path, "r") as f:
    thresholds = json.load(f)

labels = list(thresholds.keys())

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
        model = "llama-3.3-70b-versatile",
        messages= [{"role":"system","content": "You are an expert rewriter of toxic comments into non toxic ones."},{"role":"user","content":prompt}]
    )
    return response.choices[0].message.content
def predict_text(text):
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
    for i, label in enumerate(labels):
        prob = float(probs[i])
        thresh = thresholds[label]

        results[label] = {
            "confidence": prob,
            "prediction": 1 if prob >= thresh else 0
        }

        if prob > thresh:
            flagged = True
    if flagged == True:
        rewritten_text = groq_llm(text)
        return results,rewritten_text
    return results