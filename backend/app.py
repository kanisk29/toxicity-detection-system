from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import torch
from core.logging import logger

import json
logger.info('Starting FastAPI app')
app = FastAPI()

MODEL_PATH = "kanisk29/toxicity-detector-v1"
try:
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    logger.info("Model loaded successfully")

except Exception:
    logger.exception("Model loading failed")
    raise

threshold_path = hf_hub_download(
    repo_id=MODEL_PATH,
    filename="thresholds.json"
)

with open(threshold_path, "r") as f:
    thresholds = json.load(f)

labels = list(thresholds.keys())
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(req: TextRequest):
    import time
    start = time.time()
    logger.info("Received prediction request")
    logger.info(f"Received Text: {(req.text)}")
    logger.info(f"Input length: {len(req.text)}")
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

    results = {}

    for i, label in enumerate(labels):
        prob = float(probs[i])
        thresh = thresholds[label]

        results[label] = {
            "confidence": prob,
            "prediction": 1 if prob >= thresh else 0
        }
    end = time.time()
    logger.info(f"Inference time: {end - start:.4f} sec")
    logger.info("Prediction completed successfully")
    return {"results": results}
