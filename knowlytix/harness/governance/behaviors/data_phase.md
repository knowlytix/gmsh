# Data Phase Behavior Specification

## Phase: Data (Order 1)

| Step | Tool | Precedes | Requires | Conditions | On Failure |
|---|---|---|---|---|---|
| load_data | load_dataset | preprocess_data | (none) | Dataset name provided | Abort workflow |
| preprocess_data | preprocess_data | feature_select, train_model | load_data | Data loaded successfully | Retry with defaults |
| feature_select | feature_select | train_model | preprocess_data | Feature count exceeds 50 | Skip (optional step) |

## Constraints

- No model training before preprocessing is complete.
- Feature selection is optional but must occur after preprocessing if used.
- Missing value imputation is mandatory if missing percentage exceeds 5 percent.
