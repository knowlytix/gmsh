# Data Science Best Practices for Governed Agents

## Model Selection Guidelines

Guidelines for choosing the right model based on data characteristics and requirements.

### Classification Models

| Data Profile | Recommended Model | Rationale |
|---|---|---|
| High-dimensional, many categoricals | GradientBoosting | Handles categorical features natively, built-in regularization |
| High-dimensional, many categoricals | XGBoost | Handles categorical features natively, built-in regularization |
| High-dimensional, many categoricals | RandomForest | Ensemble averaging reduces variance, handles mixed types |
| Small dataset below 1000 rows | LogisticRegression | Low variance, closed-form solution, stable on small samples |
| Small dataset below 1000 rows | SVM | Kernel trick effective in low-sample regime |
| Need interpretability | LogisticRegression | Transparent coefficients, direct feature attribution |
| Need interpretability | DecisionTree | Human-readable rules, depth should be 3 or less |
| Imbalanced classes | GradientBoosting | Supports sample weighting and focal loss |
| Imbalanced classes | RandomForest | Class weight adjustment, bootstrap sampling |
| High cardinality features | GradientBoosting | Handles many levels without explosion |
| Binary target | LogisticRegression | Natural probability calibration via sigmoid |
| Binary target | GradientBoosting | High discriminative power on tabular data |
| Multiclass target | RandomForest | Native multiclass support |
| Multiclass target | GradientBoosting | One-vs-rest or softmax extension |

### Regression Models

| Data Profile | Recommended Model | Rationale |
|---|---|---|
| Linear relationships | LinearRegression | Optimal under linearity assumption |
| Linear relationships | Ridge | L2 regularization prevents overfitting |
| Nonlinear relationships | GradientBoosting | Captures complex interactions |
| Nonlinear relationships | RandomForest | Robust to outliers and nonlinearity |
| Need interpretability | LinearRegression | Direct coefficient interpretation |
| Need interpretability | DecisionTree | Depth 3 or less for transparency |
| High-dimensional sparse | Lasso | L1 sparsity for automatic feature selection |
| High-dimensional sparse | ElasticNet | Combines L1 and L2 for grouped features |

### Model Capability Matrix

| Model | Handles Categoricals | Handles Missing | Interpretable | Regularized |
|---|---|---|---|---|
| LogisticRegression | No | No | Yes | Yes (L1/L2) |
| DecisionTree | Yes | Yes | Yes (depth<=3) | No |
| RandomForest | Yes | Yes | No | Yes (ensemble) |
| GradientBoosting | Yes | Yes | No | Yes (shrinkage) |
| XGBoost | Yes | Yes | No | Yes (L1/L2/shrinkage) |
| SVM | No | No | No | Yes (C parameter) |
| LinearRegression | No | No | Yes | No |
| Ridge | No | No | Yes | Yes (L2) |
| Lasso | No | No | Yes | Yes (L1) |
| ElasticNet | No | No | Yes | Yes (L1+L2) |

## Preprocessing Requirements

Preprocessing steps depend on data conditions and the target model.

### Required Preprocessing Actions

| Condition | Required Action | Affected Models | Priority |
|---|---|---|---|
| Missing values above 5 percent | Imputation before training | All models | Critical |
| Missing values below 5 percent | Imputation or row removal | All models | High |
| Categorical features present | Encoding ordinal or one-hot | LogisticRegression, SVM, LinearRegression | Critical |
| Categorical features present | Encoding optional for tree models | DecisionTree, RandomForest, GradientBoosting | Low |
| Scale-sensitive model selected | StandardScaler or MinMaxScaler | LogisticRegression, SVM, KNN | Critical |
| Outliers detected above 3 sigma | Winsorize or clip | LinearRegression, LogisticRegression | High |
| Class imbalance ratio above 3 to 1 | SMOTE or class weights | All classification models | High |
| Feature count exceeds 50 | Feature selection recommended | All models | Medium |
| Correlated features above 0.90 | Remove or combine redundant features | LinearRegression, LogisticRegression | Medium |

### Imputation Methods

| Data Type | Missing Pattern | Recommended Method |
|---|---|---|
| Numeric continuous | Random missing | Mean or median imputation |
| Numeric continuous | Systematic missing | Model-based imputation (IterativeImputer) |
| Categorical | Random missing | Mode imputation |
| Categorical | Systematic missing | Add missing indicator category |
| Mixed types | Any pattern | ColumnTransformer with type-specific imputers |

### Scaling Methods

| Method | When to Use | Range |
|---|---|---|
| StandardScaler | General purpose, Gaussian-like features | Mean 0, Std 1 |
| MinMaxScaler | Bounded features, neural networks | 0 to 1 |
| RobustScaler | Features with outliers | Median-centered, IQR-scaled |
| No scaling needed | Tree-based models | Original scale |

## Evaluation Standards

Performance thresholds depend on the model's risk tier.

### Classification Thresholds by Risk Tier

| Risk Tier | Minimum AUC | Minimum F1 | Minimum Precision | Minimum Recall | Required Checks |
|---|---|---|---|---|---|
| High-risk (credit, lending) | 0.85 | 0.80 | 0.75 | 0.75 | Fairness, robustness, explanation, cross-validation |
| Medium-risk (marketing, operations) | 0.75 | 0.70 | 0.65 | 0.65 | Accuracy, explanation, cross-validation |
| Low-risk (internal analytics) | 0.65 | 0.60 | 0.55 | 0.55 | Accuracy only |

### Regression Thresholds by Risk Tier

| Risk Tier | Maximum RMSE Ratio | Minimum R-squared | Required Checks |
|---|---|---|---|
| High-risk | 0.15 | 0.80 | Robustness, explanation, residual analysis |
| Medium-risk | 0.25 | 0.65 | Accuracy, explanation |
| Low-risk | 0.35 | 0.50 | Accuracy only |

### Overfitting Detection

| Metric | Threshold | Interpretation |
|---|---|---|
| Train-test AUC gap | Above 0.05 | Possible overfitting, investigate |
| Train-test AUC gap | Above 0.10 | Likely overfitting, must remediate |
| Cross-validation std | Above 0.05 | Unstable model, increase regularization |
| Learning curve plateau | Not reached | Insufficient training data |

### Performance Comparison Rules

- A model must beat the baseline by at least 2 percentage points to justify added complexity.
- When two models have comparable accuracy (within 1 percentage point), prefer the more interpretable model.
- Cross-validated performance is more reliable than single train-test split.
- Report confidence intervals for all key metrics.

## Fairness Requirements

Fairness evaluation is mandatory for high-risk and medium-risk models.

### Fairness Metrics and Thresholds

| Metric | Threshold | Regulation | Applicability |
|---|---|---|---|
| Adverse Impact Ratio (AIR) | At least 0.80 | ECOA and Fair Lending | All protected attributes |
| Demographic Parity Difference | At most 0.10 | General best practice | All protected attributes |
| Equalized Odds Difference | At most 0.10 | General best practice | All protected attributes |
| False Positive Rate Difference | At most 0.05 | Fair lending scrutiny | Credit and lending models |
| Predictive Parity Difference | At most 0.10 | General best practice | All protected attributes |

### Protected Attributes

| Attribute | Category | Must Test |
|---|---|---|
| Race | Demographic | Yes for high-risk |
| Sex | Demographic | Yes for high-risk |
| Age | Demographic | Yes for high-risk |
| National origin | Demographic | Yes for credit models |
| Marital status | Demographic | Case dependent |

### Fairness Remediation

- If AIR is below 0.80, the model must not be deployed until remediated.
- Remediation options include resampling, threshold adjustment and feature removal.
- Post-remediation, all fairness metrics must be recalculated.
- Document the remediation approach and its impact on accuracy.

## Deployment Criteria

Deployment gates must be passed in order before production release.

### Gate Requirements

| Gate | Condition | Gate Type | Risk Tiers |
|---|---|---|---|
| Accuracy gate | Model accuracy above risk-tier minimum | Must pass | All tiers |
| Fairness gate | All fairness metrics within thresholds | Must pass | High-risk and medium-risk |
| Robustness gate | Performance drop under noise within 5 percent | Must pass | High-risk |
| Protected attribute gate | No untested protected attributes | Must pass | High-risk and medium-risk |
| Explanation gate | All explanation methods completed | Should pass | High-risk |
| Cross-validation gate | Cross-validated performance stable | Should pass | All tiers |
| Baseline comparison gate | Model beats baseline | Advisory | All tiers |
| Documentation gate | All diagnostics documented | Must pass | High-risk |

### Release Decision Matrix

| Gates Passed | Release Decision |
|---|---|
| All must-pass gates | Approved for deployment |
| All must-pass except one should-pass | Approved with conditions |
| Any must-pass gate failed | Blocked, remediation required |
| Advisory gates failed only | Approved with documentation |

## Workflow Ordering

The data science workflow follows a strict ordering. Each step has prerequisites.

### Step Dependencies

| Step | Precedes | Requires | Phase |
|---|---|---|---|
| load_data | preprocess_data | (none) | Data |
| preprocess_data | feature_select | load_data | Data |
| preprocess_data | train_model | load_data | Data |
| feature_select | train_model | preprocess_data | Data |
| train_model | diagnose_accuracy | preprocess_data | Training |
| train_model | predict | preprocess_data | Training |
| diagnose_accuracy | diagnose_robustness | train_model | Diagnostics |
| diagnose_accuracy | compare_models | train_model | Diagnostics |
| diagnose_robustness | diagnose_fairness | diagnose_accuracy | Diagnostics |
| diagnose_fairness | explain_importance | train_model | Diagnostics |
| explain_importance | explain_pdp | train_model | Explanation |
| explain_importance | explain_shap | train_model | Explanation |
| compare_models | release_gate | diagnose_accuracy | Evaluation |
| release_gate | deploy | compare_models | Deployment |

### Phase Ordering

| Phase | Order | Description |
|---|---|---|
| Data | 1 | Load, explore, preprocess and select features |
| Training | 2 | Train one or more candidate models |
| Diagnostics | 3 | Evaluate accuracy, robustness and fairness |
| Explanation | 4 | Generate feature importance, PDP and SHAP explanations |
| Evaluation | 5 | Compare models and select the best candidate |
| Deployment | 6 | Apply release gates and deploy approved models |

### Workflow Constraints

- No model training before preprocessing is complete.
- No diagnostics before at least one model is trained.
- No fairness evaluation before accuracy evaluation.
- No deployment before all required gates pass.
- Feature selection is optional but must occur after preprocessing if used.
- Multiple models may be trained and compared before deployment.
- Explanation steps require a trained model but can run in parallel.
