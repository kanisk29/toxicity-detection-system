# Toxicity Detection System

End-to-end multi-label toxicity classification system with real-time inference and deployment.
This project compares traditional deep learning architectures (RNN, CNN) with Transformer-based models and deploys the best-performing model in a production-ready pipeline.

Improved Macro F1 from 0.4397 → 0.6947 using per-label threshold optimization in a multi-label setting.

## Live Demo

https://toxicity-detector-by-kanisk.netlify.app  

---

## Key Contributions

- Implemented per-label threshold calibration instead of fixed 0.5 threshold  
- Achieved significant Macro F1 improvement (0.4397 → 0.6947)  
- Designed end-to-end ML system (model → API → frontend → deployment)
- Optimized training using FP16 (mixed precision) to improve GPU efficiency and reduce memory usage  
- Compared multiple architectures to establish strong baselines  

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

--- 

## Design Decisions

### Why Transformer over RNN and CNN

#### RNN (BiLSTM, BiGRU)
- Struggles with long-range dependencies in text
- Fails to capture full context of sentences with multiple clauses
- Observed issue: lower recall on minority classes such as `threat` and `identity_hate`

#### CNN
- Focuses on local n-gram patterns rather than full sentence meaning
- Faster inference but weaker semantic understanding
- Observed issue: misclassification of context-dependent toxicity

#### Transformer (RoBERTa)
- Uses self-attention to model relationships across the entire sentence
- Captures contextual meaning more effectively
- Achieved highest Macro F1, especially on minority classes

### Model Selection Tradeoff

| Model        | Accuracy | Latency | Decision |
|-------------|---------|--------|---------|
| CNN         | Low     | Very Low | Rejected due to poor contextual understanding |
| RNN         | Medium  | Medium  | Rejected due to weak minority class performance |
| RoBERTa     | High    | High    | Selected for best overall performance |

Final decision prioritized classification quality over latency for initial deployment.

---

### Per-label Performance Improvement in Transformers

| Label          | Before | After |
|----------------|-------|-------|
| toxic          | 0.816 | 0.835 |
| severe_toxic   | 0.488 | 0.552 |
| obscene        | 0.817 | 0.840 |
| threat         | 0.563 | 0.623 |
| insult         | 0.750 | 0.777 |
| identity_hate  | 0.533 | 0.539 |

---

## Final Deployed Model

- Model: RoBERTa (roberta-base), selected after empirical comparison with RNN and CNN baselines
- Task: Multi-label classification (6 labels)
- Loss Function: BCEWithLogitsLoss (with class weighting)
- Optimization: Per-label threshold tuning

---

## System Architecture

This system follows a decoupled architecture separating frontend, API, and model inference layers.

Netlify Frontend (React UI)  
→ Sends POST request to API endpoint (`/predict`)  
→ HuggingFace Space Backend (FastAPI) receives request  
→ Input text is tokenized using RoBERTa tokenizer  
→ Fine-tuned RoBERTa model performs inference (6-label sigmoid outputs)  
→ Per-label thresholding applied using precomputed thresholds (`thresholds.json`)  
→ Returns probability scores and binary predictions as JSON response  
→ (In progress) Structured logging for request tracking, inference monitoring, and debugging  

---

## Inference Flow and System Behavior

1. Client sends input text to `/predict`
2. FastAPI backend receives request
3. Text is tokenized using RoBERTa tokenizer
4. Model performs forward pass to generate logits
5. Sigmoid activation converts logits to probabilities
6. Per-label thresholds applied for final predictions
7. JSON response returned to frontend

### Current Characteristics

- Stateless API design
- Single request per inference
- No caching or batching

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
### Response [Example]

```json
{
  "toxic": 0.82,
  "severe_toxic": 0.12,
  "obscene": 0.76,
  "threat": 0.05,
  "insult": 0.64,
  "identity_hate": 0.02
}
```

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

## Key Insights

- Transformer-based modeling significantly outperforms all RNN and CNN baselines.
- Class imbalance handling is essential for multi-label toxicity detection.
- Fixed thresholds lead to suboptimal performance; per-label calibration improves Macro F1.
- Traditional models provide useful baselines but fail to capture contextual semantics as effectively as Transformers.

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

## Threshold Optimization Analysis

Default threshold of 0.5 leads to suboptimal performance in multi-label classification due to class imbalance.

### Observations

- Different labels have different probability distributions
- Rare classes such as `threat` and `identity_hate` require lower thresholds to improve recall
- Frequent classes perform better with higher thresholds to maintain precision

### Approach

- Per-label thresholds were tuned using validation data
- Objective: maximize Macro F1 score instead of individual accuracy

### Impact

- Baseline Macro F1: 0.4397
- After optimization: 0.6947

### Key Insight

A single global threshold introduces bias toward majority classes.  
Per-label calibration allows balanced performance across all categories.

---
## Project Structure

/frontend → Netlify UI  
/backend → HuggingFace Space API  
/model → Saved model and thresholds  
/notebooks → Training and experimentation  

---

## Limitations

- Performance on rare classes remains challenging  
- No multilingual support  
- Inference latency depends on HuggingFace resources  

---

## System Limitations and Tradeoffs

### Latency vs Accuracy

- RoBERTa provides strong performance but increases inference time
- Deployment on HuggingFace Spaces introduces cold-start delays

### Rare Class Performance

- Minority classes still show lower recall despite weighting and threshold tuning
- Indicates need for better data balancing or advanced loss functions


### System Constraints

- No caching layer implemented
- No batching of requests
- Each request triggers full model inference

## Future Work

- Multilingual toxicity detection  
- Model optimization (quantization, pruning)
- Production-grade logging and monitoring system  
- Continuous UI/UX improvements based on user feedback  

---

## Recent Updates

- Redesigned and improved the UI/UX to make the application more intuitive, responsive, and user-friendly  
- Enhanced frontend flow for clearer display of predictions and smoother interaction with the model  
- Currently integrating structured logging to track API requests, model outputs, and system errors for better monitoring and debugging  

---

## Summary

This project demonstrates a complete machine learning pipeline from experimentation to deployment, with recent improvements focused on usability and system reliability. The UI/UX has been enhanced to provide a smoother and more intuitive user experience, and logging capabilities are being integrated to support monitoring, debugging, and production readiness. The system addresses real-world challenges such as class imbalance and threshold calibration while leveraging Transformer-based models for strong performance.
