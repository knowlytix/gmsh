# Comparison Phase Behavior Specification

## Phase: Evaluation (Order 5)

| Step | Tool | Precedes | Requires | Conditions | On Failure |
|---|---|---|---|---|---|
| compare_models | compare_models | release_gate | diagnose_accuracy | At least two models trained | Use single model evaluation |
| release_gate | release_gate | deploy | compare_models | All must-pass gates satisfied | Block deployment |
| deploy | deploy | (none) | release_gate | Release gate passed | Abort with audit trail |

## Constraints

- No deployment before all required gates pass.
- A model must beat the baseline by at least 2 percentage points to justify added complexity.
- When two models have comparable accuracy, prefer the more interpretable model.
