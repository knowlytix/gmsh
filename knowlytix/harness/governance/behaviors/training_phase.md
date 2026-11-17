# Training Phase Behavior Specification

## Phase: Training (Order 2)

| Step | Tool | Precedes | Requires | Conditions | On Failure |
|---|---|---|---|---|---|
| train_model | train_model | diagnose_accuracy, predict | preprocess_data | Preprocessed data available | Retry with simpler model |
| predict | predict | (none) | train_model | Trained model available | Log error and continue |

## Constraints

- No model training before preprocessing is complete.
- Multiple models may be trained and compared before deployment.
- Model type should match data characteristics per GMS recommendations.
