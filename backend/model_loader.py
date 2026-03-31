import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "kanisk29/toxicity-detector-v1"    

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

from huggingface_hub import hf_hub_download
import json

threshold_path = hf_hub_download(
    repo_id=MODEL_PATH,
    filename="thresholds.json"
)

with open(threshold_path, "r") as f:
    thresholds = json.load(f)

# Label names (adjust if needed)
labels = list(thresholds.keys())


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

    for i, label in enumerate(labels):
        prob = float(probs[i])
        thresh = thresholds[label]

        results[label] = {
            "confidence": prob,
            "prediction": 1 if prob >= thresh else 0
        }

    return results