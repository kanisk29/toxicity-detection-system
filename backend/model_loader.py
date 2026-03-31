import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/transformer_model_v1"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Load thresholds
with open(f"{MODEL_PATH}/thresholds.json", "r") as f:
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