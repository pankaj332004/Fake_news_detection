# Fake News Detection — Evaluation Results

This report tracks model performance across evaluation benchmarks on the official **LIAR** dataset (Wang, ACL 2017).

---

## 1. Baseline Model: TF-IDF + Logistic Regression

- **Architecture**: `TfidfVectorizer` (unigrams + bigrams, sublinear TF scaling, max 10,000 features) + `LogisticRegression` (`C=1.0`, `class_weight='balanced'`, L-BFGS solver)
- **Execution Target**: CPU-only, ultra-fast local inference (< 5ms per statement).
- **Benchmark Splits**:
  - Training: 10,269 statements
  - Validation: 1,284 statements
  - Test: 1,283 statements

### Summary Metrics

| Split | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation** | 60.98% | 0.6095 | 0.6100 | 0.6096 | 0.6097 | 0.6521 |
| **Test** | **60.25%** | **0.5968** | **0.6032** | **0.5966** | **0.5972** | **0.6523** |

### Test Split Confusion Matrix

| | Predicted Real (0) | Predicted Fake (1) |
| :--- | :---: | :---: |
| **Actual Real (0)** | 463 (TN) | 264 (FP) |
| **Actual Fake (1)** | 246 (FN) | 310 (TP) |

### Key Observations
1. **CPU Efficiency**: The model trains in under 2 seconds on a single CPU core and requires ~15MB of RAM for inference.
2. **Balanced Performance**: Precision (59.66%) and Recall (59.72%) are well-balanced across Real and Fake classes due to `class_weight='balanced'`.
3. **Role in End-to-End Pipeline**: Acts as the Stage 3A statistical prior ($P(\text{Fake})$ vs. $P(\text{Real})$) to seed the downstream Stage 5 decision policy.
