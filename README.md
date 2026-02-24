# Toxicity Detection System

End-to-end multi-label toxicity detection system trained on the 
Jigsaw Toxic Comment Classification dataset.

The project compares traditional deep learning architectures 
(BiGRU, BiLSTM, CNN) with Transformer-based models (DistilBERT) 
and deploys the best-performing model.

---

## Problem Statement

Given a user comment, classify it into one or more toxicity categories:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

This is a **multi-label classification problem** evaluated using ROC-AUC.

---

### Experimental Setup
- Sequence length: 200
- Vocabulary size: 100,000
- Embedding dimension: 128
- Batch size: 64
- Loss: Binary Crossentropy
- Metric: Multi-label AUC

## Day 1 – RNN Baseline Models (BiGRU vs BiLSTM)

| Model   | Train Time (2 epochs) | Best Validation AUC  | 
|---------|-----------------------|----------------------|
| BiGRU   | 107.6 sec             | 0.9633               |
| BiLSTM  | 108.2 sec             | 0.9661               |

### Observations
- Both models converged within 2 epochs.
- BiLSTM achieved slightly higher validation AUC (+0.0028).
- Validation AUC peaked at Epoch 1 for both models.
- Slight performance drop in Epoch 2 suggests early stopping would improve generalization.
- Compare inference latency across models.

## Day 2 – Optimized RNN Models (BiGRU & BiLSTM)
- Added EarlyStopping based on validation AUC to prevent overfitting.
- Added ModelCheckpoint to save the best-performing model automatically.
- Expanded evaluation metrics from only AUC to AUC + Precision + Recall.
- Implemented a separate test evaluation pipeline with manual metric computation.
- Introduced threshold tuning (0.3 instead of 0.5) for multi-label classification.
- Simplified the dense layer architecture by removing the extra Dense(256) layer.
- Compared models using additional metrics to analyze the precision–recall trade-off.
- Made the training pipeline deployment-ready by saving best_model.keras.
