import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from core.logging import logger
from huggingface_hub import hf_hub_download
import json
from core.logging import get_logger

logger = get_logger(__name__)
logger.info("App started")

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