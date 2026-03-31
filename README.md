# Toxicity Detection System

End-to-end multi-label toxicity classification system with real-time deployment.
This project compares traditional deep learning architectures (RNN, CNN) with Transformer-based models and deploys the best-performing model in a production-ready pipeline.

---

## Live Demo

https://toxicity-detector-by-kanisk.netlify.app  

## Final Deployed Model

- Model: RoBERTa (roberta-base)
- Task: Multi-label classification (6 labels)
- Loss Function: BCEWithLogitsLoss (with class weighting)
- Optimization: Per-label threshold tuning

---

## Model Comparison (Macro F1)

| Model                  | Imbalance Handling | Macro F1 (Test) | Notes |
|------------------------|-------------------|------------------|------|
| BiGRU (GloVe)          | No                | —                | Poor minority recall |
| BiLSTM (GloVe)         | No                | —                | Slight precision advantage |
| BiGRU (Weighted)       | Yes               | 0.4393           | Strongest RNN baseline |
| BiLSTM (Weighted)      | Yes               | 0.3695           | High recall, low precision |
| CNN (Weighted)         | Yes               | 0.3596           | Fastest model |
| RoBERTa (Final Model)  | Yes               | 0.6947           | Best overall performance |

---

## Final Performance 

- Baseline Macro F1: 0.4397  
- Optimized Macro F1: 0.6947  

### Per-label Performance Improvement in Transformers

| Label           | Before | After |
|----------------|--------|------|
| toxic          | 0.816 |0.835 |
| severe_toxic   | 0.488 | 0.552 |
| obscene        | 0.817 | 0.840 |
| threat         | 0.563 | 0.623 |
| insult         | 0.750 | 0.777 |
| identity_hate  | 0.533 | 0.539 |

---

## Key Insights

- Transformer-based modeling significantly outperforms all RNN and CNN baselines.
- Class imbalance handling is essential for multi-label toxicity detection.
- Fixed thresholds lead to suboptimal performance; per-label calibration improves Macro F1.
- Traditional models provide useful baselines but fail to capture contextual semantics as effectively as Transformers.

---

## System Architecture

Frontend 
→ Sends input text via API request  
→ HuggingFace Space backend processes request  
→ Tokenization using RoBERTa tokenizer  
→ Model inference  
→ Per-label thresholding  
→ Returns probabilities and binary predictions  

---

## API Usage

### Endpoint

POST /predict

### Request

```json
{
  "text": "Your input sentence"
}
```
### Response
{
  "toxic": 0.82,
  "severe_toxic": 0.12,
  "obscene": 0.76,
  "threat": 0.05,
  "insult": 0.64,
  "identity_hate": 0.02
}

## Problem Statement

Given a user comment, classify it into one or more of the following categories:

- toxic  
- severe_toxic  
- obscene  
- threat  
- insult  
- identity_hate  

This is a multi-label classification problem evaluated using Macro F1 score.

---

## Core Techniques

### Class Imbalance Handling
- Computed label-wise positive weights  
- Applied weighted binary cross-entropy  
- Improved recall on minority classes  

### Threshold Optimization
- Per-label threshold tuning using validation data  
- Maximizes Macro F1  
- Avoids bias introduced by fixed thresholds  

### Transformer Fine-Tuning
- Pretrained RoBERTa model  
- Context-aware embeddings  
- Independent sigmoid outputs for each label  

---

## Project Structure

/frontend → Netlify UI  
/backend → HuggingFace Space API  
/model → Saved model and thresholds  
/notebooks → Training and experimentation  

---

## Limitations

- Performance on rare classes remains challenging  
- Dataset bias (Jigsaw dataset)  
- No multilingual support  
- Inference latency depends on HuggingFace resources  

---

## Future Work

- Multilingual toxicity detection  
- Model optimization (quantization, pruning)  
- Dedicated backend deployment (FastAPI, Docker)  
- Explainability methods for predictions  

---

## Summary

This project demonstrates a complete machine learning pipeline from experimentation to deployment, addressing real-world challenges such as class imbalance and threshold calibration while leveraging Transformer-based models for improved performance.
