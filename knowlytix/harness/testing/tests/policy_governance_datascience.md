# Data Science Governance Policy Corpus

**Scope:** Enterprise data-science agent fleet for high-stakes classification
and regression models. This corpus defines who may do what, which
workflows meet the engineering bar, and which do not.

**Framing.** Governance here is an engineering discipline: preventing
silent failure in segments the aggregate metrics hide, preventing
prediction decay as data drifts, and preventing effects from moving
between retrains. It does not rest on any external regulatory regime;
the bars are the organization's own.

This document is the ground-truth corpus for the data-science policy
GMS. It is organized into five layers: agents, tools, data classes,
standards, and workflows.

## 1. Agents and Roles

### Table 1.1 - Agent Catalog

| Agent ID | Role | Seniority | Authority Level | Function |
|:---------|:-----|:---------:|:---------------:|:---------|
| data_engineer | Data Engineer | 2 | 1 | ingest + schema |
| data_steward | Data Steward | 3 | 2 | classification + access |
| schema_curator | Schema Curator | 2 | 1 | schema versioning |
| eda_analyst | EDA Analyst | 1 | 1 | exploration |
| feature_engineer | Feature Engineer | 2 | 1 | feature construction |
| feature_selector | Feature Selector | 2 | 1 | selection + selection-stability |
| modeler_tree | Tree Modeler | 2 | 1 | tree-based fits |
| modeler_linear | Linear Modeler | 2 | 1 | linear/GLM fits |
| modeler_nn | Neural Modeler | 3 | 1 | neural fits |
| ensembler | Ensembler | 3 | 1 | stacking/ensembling |
| validation_reviewer | Validation Reviewer | 3 | 2 | holdout + challenger |
| weakness_reviewer | Weakness Reviewer | 3 | 2 | segment-level failure |
| resilience_reviewer | Resilience Reviewer | 3 | 2 | drift degradation |
| stability_reviewer | Stability Reviewer | 3 | 2 | effect stability |
| robustness_reviewer | Robustness Reviewer | 3 | 2 | perturbation bounds |
| fairness_reviewer | Fairness Reviewer | 3 | 2 | protected-attribute parity |
| explainability_reviewer | Explainability Reviewer | 3 | 2 | attribution coverage |
| mlops_engineer | MLOps Engineer | 3 | 2 | deploy pipeline |
| deploy_operator | Deploy Operator | 2 | 3 | push to prod |
| approval_officer | Approval Officer | 4 | 3 | tier-based sign-off |
| model_risk_officer | Model Risk Officer | 5 | 4 | oversight + policy + charter |
| incident_responder | Incident Responder | 3 | 2 | production failure |
| retrain_agent | Retrain Agent | 2 | 2 | automated retraining |
| audit_logger | Audit Logger | 2 | 1 | trace + telemetry |

## 2. Tools and Capabilities

### Table 2.1 - Tool Catalog

| Tool ID | Purpose | Risk Tier | Side Effects |
|:--------|:--------|:---------:|:-------------|
| load_dataset | Load tabular dataset | 4 | none |
| data_summary | Summarize distributions | 4 | none |
| eda_correlation | Pairwise correlation | 4 | none |
| preprocess_data | Impute + scale + encode | 3 | mutates session state |
| feature_select | Rank + subset features | 3 | mutates session state |
| train_model | Fit an estimator | 3 | mutates session state |
| predict | Score held-out rows | 3 | none |
| leaderboard | Rank trained models | 3 | none |
| diagnose_accuracy | Aggregate metrics | 3 | none |
| diagnose_weakness | Segment-level failure (RF-proximity clustering) | 2 | none, requires RandomForest |
| diagnose_resilience | Drift degradation (RF-on-errors + targeted distribution shift) | 2 | none, requires RandomForest |
| diagnose_stability | Effect + selection stability (RF-based) | 2 | none, requires RandomForest |
| diagnose_robustness | Perturbation bounds | 2 | none |
| diagnose_fairness | Protected-attribute parity | 2 | none |
| explain_pfi | Permutation importance | 3 | none |
| explain_pdp | Partial dependence | 3 | none |
| explain_shap | SHAP values | 3 | none |
| approve_model | Mark model release-ready | 1 | writes approval record |
| deploy_model | Push model to production | 1 | mutates prod state |
| trigger_retrain | Kick off retrain job | 2 | mutates training queue |

## 3. Data Classes

### Table 3.1 - Data Class Catalog

| Data Class | Sensitivity | Example Fields |
|:-----------|:-----------:|:---------------|
| public | 1 | aggregate statistics, published benchmarks |
| internal | 2 | engineered features, model scores |
| confidential | 3 | raw training records, customer identifiers hashed |
| pii | 4 | customer name, address, SSN, protected attributes |
| regulated | 5 | labeled adverse-action data, decisioned outcomes |

## 4. Standards Catalog

### Table 4.1 - Engineering Quality Bars

| Standard ID | Concern | Pass Criterion (illustrative) |
|:------------|:--------|:------------------------------|
| model_validation_standard | held-out performance | CV + challenger + back-testing |
| weakness_identification_standard | segment failure | error-aware clustering via RandomForest proximity; no discovered cluster with AUC < 0.5 or accuracy below random baseline; max cluster-performance gap bounded |
| resilience_standard | drift degradation | identify error-driving variables via RF-on-absolute-errors; under bounded distributional shifts (+/- 1 sigma on continuous, +/- 20% reweight on categorical) applied to top-k error-driving variables, primary model performance degradation bounded |
| stability_standard | effect stability | per-tree importance distribution from a RandomForest; top-k Jaccard >= 0.7, rank-Spearman >= 0.8 across trees (and across time windows when re-fit) |
| robustness_standard | perturbation stability | output change bounded under bounded input perturbation |
| fairness_standard | parity | AIR >= 0.80, equalized-odds gap bounded |
| explainability_standard | attribution coverage | global + local attributions produced and consistent |
| data_quality_standard | input hygiene | schema match, completeness, distribution-drift alert |
| privacy_standard | PII handling | no-PII in features, anonymization on export |
| retention_standard | artifact lifecycle | training data + model artifact retained per policy, disposed on time |
| change_control_standard | release discipline | versioned, rollback-capable, dual-control |
| monitoring_standard | production health | drift detection, incident response, SLA |
| approval_standard | sign-off | tier-based approval trail |

## 5. Workflows and Compliance

### Table 5.1 - Workflow Compliance Status

| Workflow | Risk Tier | Status | Description |
|:---------|:---------:|:------:|:------------|
| high_risk_classification | 1 | pass | exemplar — passes all three pillars + validation + fairness + monitoring |
| regulated_lending_model | 1 | pass | passes weakness + resilience + stability + fairness + explainability |
| clinical_triage_model | 1 | pass | passes weakness + resilience + stability + explainability + privacy |
| conservative_credit_model | 2 | pass | passes weakness + resilience + stability + validation |
| reviewed_fraud_detection | 2 | pass | passes weakness + robustness + monitoring + change_control |
| monitored_churn_model | 3 | pass | passes resilience + monitoring + data_quality |
| validated_baseline_model | 3 | pass | passes validation + data_quality + change_control |
| aggregate_only_workflow | 2 | fail | violates weakness — reports only aggregate AUC, no segment analysis |
| snapshot_only_workflow | 2 | fail | violates resilience — no drift or temporal evaluation |
| unstable_lasso_workflow | 2 | fail | violates stability — selected-features Jaccard < 0.3 across bootstraps |
| shadow_pipeline | 1 | fail | violates change_control + approval — no versioning, no sign-off |
| legacy_workflow_v1 | 1 | fail | violates validation + monitoring — pre-dates current quality bars |
| unreviewed_deployment | 1 | fail | violates approval + change_control |
| unmonitored_legacy_model | 2 | fail | violates monitoring + resilience |
| overfit_trial_model | 3 | fail | violates validation + stability |

## 6. Rule Narrative

**Tool-use authorization.** Each agent has a fixed tool authorization set
determined by function. Reviewers may use diagnostic and explanation
tools but may not train or deploy. Modelers may train and explain but
may not deploy. Deployment and approval tools are restricted to the
approval and ops roles.

**Data access.** Access to a data class is read-before-write. PII and
regulated data classes are restricted to stewards, validators and
auditors; modelers and reviewers read from the engineered-feature
(internal) class only.

**Escalation.** Each reviewer role escalates to the model_risk_officer,
which is the terminal authority. Incident response escalates first to
the mlops_engineer, then to the incident_responder, then to the
model_risk_officer.

**Approval hierarchy.** Tier-1 (restricted) actions require
approval_officer sign-off plus a co-sign from the model_risk_officer.
Tier-2 actions require approval_officer alone. Tier-3 actions are
self-service for authorized agents.

**The three pillars.** A high-stakes workflow does not count as
compliant with its engineering bar unless the agent has produced
evidence for three separable questions. Where does the model fail now
(weakness)? Does the model still predict well as data drifts
(resilience)? Do the model's effects stay put across bootstraps,
time windows, and retrains (stability)? Aggregate accuracy does not
substitute for any of the three.

**All three pillars use RandomForest as the diagnostic substrate.**
Each pillar reads a different internal structure of the forest and
projects it onto a different question.

*Weakness* is discovered via error-aware clustering on RandomForest
proximity. For held-out points $x_i$ and $x_j$, proximity is the
fraction of trees in which they land in the same terminal leaf --- a
supervised, scale-invariant, interaction-aware similarity. The
proximity is combined with the per-row prediction error so that
pairs of wrong predictions pull together, and the weighted similarity
is clustered. Each discovered cluster is a candidate weak segment,
tested against the pass criterion on its own metrics. The agent does
not nominate the slicing dimensions; the forest does.

*Stability* reads the per-tree feature-importance distribution
directly from the forest. The forest is itself a bootstrap ensemble,
so no external bootstrap loop is needed. Top-k Jaccard across trees
and Spearman rank correlation of importance rankings measure whether
the selected effects are stable or a lottery.

*Resilience* fits a second RandomForest whose target is the primary
model's absolute error. Permutation importance on this error-RF
identifies the variables that most drive where the primary model
fails. For each top-k error-driving variable, the diagnostic applies
systematic distributional shifts --- mean shift, scale expansion,
proportion reweight --- and re-scores the primary model to measure
the degradation. Stress-testing is targeted at the variables that
actually matter, not applied uniformly to all features.

A high-stakes workflow therefore must train a RandomForest --- either
as the primary model or as a dedicated challenger on the same
features and target --- before any of the three diagnostics can run.
