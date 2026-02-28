# Toxicity Detection System

End-to-end multi-label toxicity detection system.
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
  
| Model                       | Class Imbalance Handled | Best Val AUC | Best Val Precision | Best Val Recall | Macro F1 (Test) | Train Time |
|-----------------------------|---------------|--------------|--------------------|------------------|------------------|------------|
| BiGRU (Day 3 – GloVe)      |  No         | 0.9732       | 0.8183             | 0.7468           | —                | 172 sec    |
| BiLSTM (Day 3 – GloVe)     |  No         | 0.9717       | 0.8552             | 0.7216           | —                | 162 sec    |
| **BiGRU (Day 4 – Weighted)**  |  Yes        | **0.9811**   | 0.4907             | 0.9355           | **0.4393**       | 201 sec    |
| **BiLSTM (Day 4 – Weighted)** |  Yes        | 0.9808       | 0.4155             | 0.9491           | 0.3695           | 152 sec    |
| **CNN (Day 4 – Weighted)**    |  Yes        | 0.9774       | 0.4147             | **0.9717**       | 0.3596           | **69 sec** |

## Refer to this table for improvements to the model. 

# Day 1 – RNN Baseline Models (BiGRU vs BiLSTM)

### Observations
- Both models converged within 2 epochs.
- BiLSTM achieved slightly higher validation AUC (+0.0028).
- Validation AUC peaked at Epoch 1 for both models.
- Slight performance drop in Epoch 2 suggests early stopping would improve generalization.
- Compare inference latency across models.

# Day 2 – Optimized RNN Models (BiGRU & BiLSTM)

Day 2 focused on improving generalization, evaluation depth, and deployment readiness of the baseline RNN models.

- Introduced **EarlyStopping** (monitoring validation AUC) to prevent overfitting and restore the best weights.
- Added **ModelCheckpoint** to automatically save the best-performing model (`best_model.keras`).
- Expanded evaluation metrics from only AUC to **AUC + Precision + Recall** for better analysis of multi-label performance.
- Implemented a dedicated **test evaluation pipeline** with manual metric computation.
- Applied **threshold tuning (0.3 instead of 0.5)** to better balance precision–recall trade-offs.
- Simplified the dense architecture by removing the extra `Dense(256)` layer to reduce model complexity.
- Compared models across multiple metrics to analyze precision vs recall behavior.
- Made the training workflow deployment-ready by standardizing model saving and evaluation.
  
# Day 3 – GloVe Integration & Embedding Experiments

## Objective

Enhance semantic representation by replacing randomly initialized embeddings with pretrained **GloVe (100d)** vectors and compare:

- Trainable embeddings (fine-tuned)
- Frozen embeddings (static)

---

## Implementation

- Loaded `glove-wiki-gigaword-100` using `gensim`.
- Built an embedding matrix aligned with the `TextVectorization` vocabulary.
- Achieved ~58% vocabulary coverage.
- Created two embedding configurations:
  - `trainable=True`
  - `trainable=False`
- Trained four models:
  - BiGRU (Trainable)
  - BiGRU (Frozen)
  - BiLSTM (Trainable)
  - BiLSTM (Frozen)

---

## Key Observations

- Pretrained embeddings improved semantic initialization.

- ~58% coverage makes fine-tuning important.

- Frozen embeddings train slightly faster.

- Trainable embeddings generally achieve better validation AUC.

- GRU remains slightly faster than LSTM.

#### Finally, decided to move on with the model using ```trainable = True``` since it gave much better results as shown in the table. 

# Day 4 – Class Imbalance Handling and CNN Architecture

Day 4 focuses on addressing severe class imbalance in the dataset and expanding the architecture comparison by introducing a CNN-based model alongside RNN variants.

---

## Class Imbalance Analysis

The dataset is highly imbalanced across toxicity categories.  
Rare classes such as `threat`, `severe_toxic`, and `identity_hate` occur significantly less frequently than `toxic` or non-toxic samples.

Computed positive class weights:

- toxic → 5.21  
- severe_toxic → 50.02  
- obscene → 9.44  
- threat → 166.91  
- insult → 10.12  
- identity_hate → 56.78  

The extreme weight for `threat` confirms severe imbalance.  
Without correction, the model favors the dominant negative class, leading to poor minority recall despite high AUC.

---

## Weighted Binary Crossentropy

To mitigate imbalance, a custom weighted binary crossentropy loss was implemented:

Weighted BCE =  
− ( w * y_true * log(y_pred) + (1 − y_true) * log(1 − y_pred) )

Where:

- `w` represents class-specific positive weights  
- Minority labels receive stronger gradient updates  
- False negatives are penalized more heavily  

This modification shifts optimization toward recall improvement for rare categories.

---

## Threshold Adjustment

The prediction threshold was reduced from 0.5 to 0.3.

Effects observed:

- Substantial recall increase  
- Lower precision due to more aggressive predictions  
- Improved Macro F1 compared to previous versions  

This adjustment better aligns the system with safety-focused deployment scenarios.

---

## CNN Architecture Introduction

In addition to BiGRU and BiLSTM models, a CNN-based architecture was introduced.

Architecture:

- Trainable GloVe embedding layer  
- Conv1D (64 filters, kernel size = 5)  
- Conv1D (128 filters, kernel size = 5)  
- GlobalMaxPooling1D  
- Dense layers  
- Sigmoid output (6 labels)  

The CNN captures local n-gram patterns efficiently and trains significantly faster than RNN variants, providing a strong computational baseline.

---

## Key Observations

- Weighted loss significantly increased recall across all models.
- GRU achieved the best overall balance (highest Macro F1).
- CNN trained nearly 3× faster than RNNs but slightly underperformed in Macro F1.
- Precision decreased due to aggressive minority detection, which is expected under weighted optimization.
- AUC improved beyond 0.98, indicating strong ranking capability despite imbalance.
