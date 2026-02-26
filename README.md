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
  
| Model                         | Embedding Type        | Train Time (≈3 epochs) | Best Validation AUC |
|--------------------------------|------------------------|------------------------|---------------------|
| BiGRU (Day 1 Baseline)        | Random Init            | 107.6 sec (2 ep)       | 0.9633              |
| BiLSTM (Day 1 Baseline)       | Random Init            | 108.2 sec (2 ep)       | 0.9661              |
| BiGRU (GloVe)                 | Trainable = True       | 172.7 sec              | **0.9732**          |
| BiGRU (GloVe)                 | Trainable = False      | 210.0 sec              | 0.9720              |
| BiLSTM (GloVe)                | Trainable = True       | 162.5 sec              | **0.9717**          |
| BiLSTM (GloVe)                | Trainable = False      | 254.7 sec              | 0.9716              |
| CNN                           | —                      | —                      | —                   |
| DistilBERT                    | —                      | —                      | —                   |

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

## Model Update

```python
Embedding(
    input_dim=len(vocab),
    output_dim=100,
    weights=[embedding_matrix],
    mask_zero=True,
    trainable=True/False
)
```
## Key Observations

- Pretrained embeddings improved semantic initialization.

- ~58% coverage makes fine-tuning important.

- Frozen embeddings train slightly faster.

- Trainable embeddings generally achieve better validation AUC.

- GRU remains slightly faster than LSTM.

#### Finally, decided to move on with the model using ```trainable = True``` since it gave much better results as shown in the table. 
