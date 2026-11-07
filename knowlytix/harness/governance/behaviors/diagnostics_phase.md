# Diagnostics Phase Behavior Specification

## Phase: Diagnostics (Order 3)

| Step | Tool | Precedes | Requires | Conditions | On Failure |
|---|---|---|---|---|---|
| diagnose_accuracy | diagnose_accuracy | diagnose_robustness, compare_models | train_model | Trained model available | Log failure, block deployment |
| diagnose_robustness | diagnose_robustness | diagnose_fairness | diagnose_accuracy | Accuracy diagnosed | Log failure, block deployment |
| diagnose_fairness | diagnose_fairness | explain_importance | diagnose_robustness | Robustness diagnosed | Log failure, block deployment |

## Constraints

- No diagnostics before at least one model is trained.
- No fairness evaluation before accuracy evaluation.
- Robustness evaluation required for high-risk models.
