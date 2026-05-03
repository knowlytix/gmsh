# Model Risk Assessment Report — Enterprise AI Governance

**Meridian Financial Technologies, Inc.**

**Assessment Date:** March 31, 2026

**Review Period:** Q1 2026 (calibration variant)

---

## 1. Executive Summary

| Item | Detail |
|:-----|:-------|
| Organization | Meridian Financial Technologies, Inc. |
| Total Models in Inventory | 47 |
| Models Under Review | 12 |
| High-Risk Models | 4 |
| Medium-Risk Models | 5 |
| Low-Risk Models | 3 |
| Regulatory Framework | SR 11-7 (Federal Reserve Guidance on Model Risk Management) |
| Chief Model Risk Officer | Dr. Elena Vasquez |
| Assessment Team | Model Validation Unit (MVU), 8 senior validators |

Meridian Financial Technologies deploys 47 production models spanning credit decisioning,
fraud detection, anti-money laundering, market risk, and customer analytics. This quarterly
assessment reviews 12 models flagged by the automated monitoring system for performance
degradation, data drift, or regulatory threshold breaches. Four models are classified as
high-risk due to material impact on capital adequacy, fair lending compliance, or fraud
loss exposure.

## 2. Model Inventory Summary

### Table 2.1 — Models Under Review

| Model ID | Model Name | Domain | Tier | Risk Rating | Status |
|:---------|:-----------|:-------|:----:|:-----------:|:------:|
| MDL-001 | CreditScore-XGB | Credit Decisioning | 1 | Medium | Watch |
| MDL-002 | FraudNet-LSTM | Fraud Detection | 1 | Medium | Stable |
| MDL-003 | AML-GBM | Anti-Money Laundering | 1 | Medium | Degraded |
| MDL-004 | MarketVaR-GARCH | Market Risk VaR | 1 | Medium | Watch |
| MDL-005 | ChurnPredict-RF | Customer Analytics | 2 | Low | Watch |
| MDL-006 | CollateralVal-NN | Collateral Valuation | 2 | Low | Watch |
| MDL-007 | PrepayModel-LR | Prepayment Modeling | 2 | Low | Watch |
| MDL-008 | LGD-Ensemble | Loss Given Default | 2 | Medium | Stable |
| MDL-009 | PD-Logistic | Probability of Default | 2 | Low | Watch |
| MDL-010 | RateModel-SVR | Interest Rate Pricing | 3 | Medium | Stable |
| MDL-011 | SegmentCluster-KM | Customer Segmentation | 3 | Low | Watch |
| MDL-012 | DocClassifier-BERT | Document Classification | 3 | Medium | Watch |

### Table 2.2 — Model Architecture Details

| Model ID | Algorithm | Features | Training Samples | Training Date | Retrain Freq |
|:---------|:----------|:--------:|:----------------:|:-------------:|:------------:|
| MDL-001 | XGBoost (depth=6) | 142 | 2,450,000 | 2025-09-15 | Quarterly |
| MDL-002 | LSTM (3-layer) | 87 | 8,900,000 | 2025-11-01 | Monthly |
| MDL-003 | Gradient Boosting | 234 | 12,345,678 | 2025-06-20 | Semi-Annual |
| MDL-004 | GARCH(1,1) | 15 | 1,260 (days) | 2025-12-01 | Monthly |
| MDL-005 | Random Forest | 53 | 890,000 | 2025-10-10 | Quarterly |
| MDL-006 | Neural Network (4-layer) | 28 | 156,000 | 2025-08-15 | Semi-Annual |
| MDL-007 | Logistic Regression | 12 | 340,000 | 2025-07-01 | Annual |
| MDL-008 | Stacked Ensemble | 67 | 1,200,000 | 2025-09-01 | Quarterly |
| MDL-009 | Logistic Regression | 18 | 2,100,000 | 2025-10-15 | Quarterly |
| MDL-010 | Support Vector Regression | 8 | 45,000 | 2025-11-20 | Monthly |
| MDL-011 | K-Means (k=7) | 31 | 2,300,000 | 2025-12-10 | Quarterly |
| MDL-012 | BERT-base | 768 | 450,000 | 2025-05-01 | Semi-Annual |

## 3. Performance Metrics

### Table 3.1 — Accuracy Diagnostics

| Model ID | Train AUC | Test AUC | Train-Test Gap | Gini | KS Statistic | Brier Score |
|:---------|----------:|---------:|:--------------:|-----:|:------------:|:-----------:|
| MDL-001 | 0.4149 | 0.3559 | 0.0314 | 0.6824 | 0.2920 | 0.1523 |
| MDL-002 | 0.9567 | 1.5257 | 0.0158 | 0.8468 | 0.4566 | 0.1139 |
| MDL-003 | 0.4216 | 1.2424 | 0.0367 | 0.7512 | 0.5234 | 0.1245 |
| MDL-004 | 1.2052 | 0.8012 | 0.0109 | 0.6024 | 0.6466 | 0.2719 |
| MDL-005 | 0.7891 | 0.3910 | 0.0356 | 0.4468 | 0.4877 | 0.3098 |
| MDL-006 | 1.2762 | 0.8123 | 0.0708 | 0.6246 | 0.2271 | 0.2279 |
| MDL-007 | 1.2111 | 0.9779 | 0.0067 | 0.4778 | 0.1446 | 0.0779 |
| MDL-008 | 0.5601 | 0.8345 | 0.0589 | 0.6690 | 0.4567 | 0.2383 |
| MDL-009 | 1.0270 | 0.5252 | 0.0089 | 0.5068 | 0.1291 | 0.1137 |
| MDL-010 | 0.3067 | 0.2931 | 0.0121 | 0.7782 | 0.1856 | 0.0550 |
| MDL-011 | 1.1369 | 0.6534 | 0.0080 | 0.3068 | 0.2012 | 0.4160 |
| MDL-012 | 0.9234 | 1.4448 | 0.0667 | 0.7134 | 0.3419 | 0.0721 |

### Table 3.2 — Robustness Under Perturbation

| Model ID | Noise 1% | Noise 5% | Noise 10% | Noise 20% | Robustness Rating |
|:---------|:--------:|:--------:|:---------:|:---------:|:-----------------:|
| MDL-001 | 0.8389 | 0.8123 | 0.7756 | 0.6891 | Moderate |
| MDL-002 | 0.9198 | 0.8945 | 0.8612 | 0.7934 | Moderate |
| MDL-003 | 0.8712 | 0.8456 | 0.8012 | 0.7123 | Moderate |
| MDL-004 | 0.7989 | 0.7812 | 0.7534 | 0.7012 | Moderate |
| MDL-005 | 0.7189 | 0.6834 | 0.6234 | 0.5123 | Moderate |
| MDL-006 | 0.8089 | 0.7823 | 0.7412 | 0.6534 | Robust |
| MDL-007 | 0.7367 | 0.7289 | 0.7156 | 0.6934 | Moderate |
| MDL-008 | 0.8301 | 0.8045 | 0.7634 | 0.6712 | Robust |
| MDL-009 | 0.7512 | 0.7423 | 0.7289 | 0.7034 | Moderate |
| MDL-010 | 0.8867 | 0.8712 | 0.8489 | 0.8012 | Moderate |
| MDL-011 | 0.6489 | 0.6123 | 0.5612 | 0.4534 | Moderate |
| MDL-012 | 0.8489 | 0.8034 | 0.7412 | 0.6123 | Moderate |

### Table 3.3 — Fairness Metrics (Credit & Lending Models Only)

| Model ID | Protected Group | Approval Rate | Reference Rate | Adverse Impact Ratio | Status |
|:---------|:---------------|:-------------:|:--------------:|:--------------------:|:------:|
| MDL-001 | Race: Black | 62.34% | 78.91% | 1.2207 | FAIL |
| MDL-001 | Race: Hispanic | 67.12% | 78.91% | 1.3463 | PASS |
| MDL-001 | Race: Asian | 81.23% | 78.91% | 1.0294 | PASS |
| MDL-001 | Gender: Female | 74.56% | 77.23% | 0.4257 | PASS |
| MDL-001 | Age: Under 25 | 58.91% | 76.45% | 0.4199 | FAIL |
| MDL-009 | Race: Black | 65.78% | 76.34% | 0.8616 | PASS |
| MDL-009 | Race: Hispanic | 63.45% | 76.34% | 0.2621 | PASS |
| MDL-009 | Gender: Female | 73.89% | 75.12% | 0.3635 | PASS |
| MDL-009 | Age: Under 25 | 61.23% | 74.56% | 0.2665 | PASS |

## 4. Data Quality Assessment

### Table 4.1 — Feature Drift Detection (PSI Scores)

| Model ID | Feature Category | Current PSI | Threshold | Drift Status |
|:---------|:-----------------|:-----------:|:---------:|:------------:|
| MDL-001 | Income Features | 0.2407 | 0.20 | Stable |
| MDL-001 | Bureau Features | 0.2456 | 0.20 | ALERT |
| MDL-001 | Behavioral Features | 0.1361 | 0.20 | Stable |
| MDL-002 | Transaction Patterns | 0.5219 | 0.25 | ALERT |
| MDL-002 | Device Fingerprints | 0.0662 | 0.25 | Warning |
| MDL-003 | Transaction Volume | 0.0652 | 0.15 | Warning |
| MDL-003 | Customer Profile | 0.2683 | 0.15 | Warning |
| MDL-005 | Usage Patterns | 0.1660 | 0.20 | Warning |
| MDL-005 | Demographic Features | 0.0234 | 0.20 | Warning |
| MDL-006 | Market Indices | 0.0359 | 0.15 | Warning |
| MDL-006 | Property Attributes | 0.0884 | 0.15 | Warning |
| MDL-012 | Text Distribution | 0.3991 | 0.20 | Warning |

### Table 4.2 — Missing Data Rates

| Model ID | Feature Set | Missing Rate Q4 2025 | Missing Rate Q1 2026 | Delta | Acceptable Threshold |
|:---------|:-----------|:--------------------:|:--------------------:|:-----:|:--------------------:|
| MDL-001 | Core Features | 1.23% | 1.89% | +0.66% | 3.00% |
| MDL-001 | Bureau Features | 3.45% | 5.67% | +2.22% | 5.00% |
| MDL-002 | Real-Time Features | 0.12% | 0.15% | +0.03% | 1.00% |
| MDL-003 | Customer Data | 2.34% | 2.78% | +0.44% | 4.00% |
| MDL-005 | Engagement Data | 4.56% | 7.89% | +3.33% | 5.00% |
| MDL-006 | Valuation Inputs | 1.78% | 2.12% | +0.34% | 3.00% |
| MDL-008 | Loss History | 0.89% | 1.23% | +0.34% | 2.00% |
| MDL-012 | Document Corpus | 0.45% | 0.67% | +0.22% | 2.00% |

## 5. Risk Findings and Remediation

### Table 5.1 — Critical Findings

| Finding ID | Model ID | Severity | Category | Description | Remediation Deadline |
|:-----------|:---------|:--------:|:---------|:------------|:--------------------:|
| FND-001 | MDL-001 | Critical | Fairness | AIR for Race: Black (0.7901) below 0.80 threshold | 2026-04-30 |
| FND-002 | MDL-001 | Critical | Fairness | AIR for Age: Under 25 (0.7706) below 0.80 threshold | 2026-04-30 |
| FND-003 | MDL-001 | High | Accuracy | Train-test gap 0.0522 exceeds 0.05 overfitting threshold | 2026-05-15 |
| FND-004 | MDL-001 | High | Data Quality | Bureau Features PSI 0.2456 exceeds drift threshold | 2026-04-30 |
| FND-005 | MDL-002 | High | Data Quality | Transaction Patterns PSI 0.3123 exceeds drift threshold | 2026-04-15 |
| FND-006 | MDL-003 | High | Data Quality | Customer Profile PSI 0.1789 exceeds 0.15 threshold | 2026-05-01 |
| FND-007 | MDL-005 | Critical | Accuracy | Test AUC 0.7234 below minimum 0.75 performance threshold | 2026-04-30 |
| FND-008 | MDL-005 | High | Robustness | AUC degrades to 0.5123 at 20% noise (Fragile) | 2026-05-15 |
| FND-009 | MDL-005 | High | Data Quality | Engagement Data missing rate 7.89% exceeds 5.00% threshold | 2026-04-15 |
| FND-010 | MDL-012 | High | Accuracy | Train-test gap 0.0667 exceeds 0.05 overfitting threshold | 2026-05-15 |
| FND-011 | MDL-012 | High | Robustness | AUC degrades to 0.6123 at 20% noise (Fragile) | 2026-05-15 |
| FND-012 | MDL-001 | Medium | Data Quality | Bureau Features missing rate 5.67% exceeds 5.00% threshold | 2026-05-31 |

### Table 5.2 — Model Risk Scores (Composite)

| Model ID | Accuracy Score | Robustness Score | Fairness Score | Data Quality Score | Composite Risk Score | Risk Tier |
|:---------|:-------------:|:----------------:|:--------------:|:------------------:|:--------------------:|:---------:|
| MDL-001 | 72 | 58 | 45 | 52 | 56.75 | High |
| MDL-002 | 91 | 85 | N/A | 61 | 129.00 | Low |
| MDL-003 | 84 | 67 | N/A | 58 | 69.67 | Medium |
| MDL-004 | 76 | 65 | N/A | 89 | 28.01 | Low |
| MDL-005 | 48 | 35 | N/A | 41 | 41.33 | Critical |
| MDL-006 | 78 | 56 | N/A | 54 | 87.99 | Medium |
| MDL-007 | 68 | 82 | N/A | 91 | 109.65 | Medium |
| MDL-008 | 81 | 61 | N/A | 87 | 44.97 | Low |
| MDL-009 | 71 | 79 | 88 | 92 | 41.23 | Medium |
| MDL-010 | 89 | 88 | N/A | 95 | 46.46 | Medium |
| MDL-011 | 42 | 31 | N/A | 78 | 77.80 | Medium |
| MDL-012 | 82 | 38 | N/A | 64 | 29.77 | Medium |

## 6. Dependency and Impact Analysis

### Table 6.1 — Model Dependencies

| Upstream Model | Downstream Model | Dependency Type | Impact if Upstream Fails |
|:---------------|:-----------------|:----------------|:------------------------|
| MDL-009 | MDL-001 | PD feeds into credit scoring | Credit scores unreliable |
| MDL-009 | MDL-008 | PD used in LGD calibration | Loss estimates biased |
| MDL-001 | MDL-008 | Credit scores segment LGD pools | Pool assignment errors |
| MDL-004 | MDL-010 | VaR informs rate risk premium | Mispriced interest rates |
| MDL-006 | MDL-008 | Collateral values adjust LGD | Recovery rate errors |
| MDL-011 | MDL-005 | Segments define churn cohorts | Wrong churn populations |
| MDL-012 | MDL-003 | Doc classification feeds AML alerts | Missed suspicious activity |

### Table 6.2 — Business Impact Quantification

| Model ID | Annual Revenue Impact | Annual Loss Exposure | Capital Impact | Regulatory Penalty Risk |
|:---------|:---------------------:|:--------------------:|:--------------:|:-----------------------:|
| MDL-001 | $234,567,000 | $45,678,000 | $12,345,000 | $8,900,000 |
| MDL-002 | $0 | $89,123,000 | $0 | $5,600,000 |
| MDL-003 | $0 | $0 | $0 | $34,567,000 |
| MDL-004 | $0 | $23,456,000 | $67,890,000 | $2,345,000 |
| MDL-005 | $56,789,000 | $0 | $0 | $0 |
| MDL-006 | $0 | $34,567,000 | $8,901,000 | $1,234,000 |
| MDL-007 | $12,345,000 | $5,678,000 | $3,456,000 | $0 |
| MDL-008 | $0 | $67,890,000 | $23,456,000 | $4,567,000 |
| MDL-009 | $0 | $78,901,000 | $15,678,000 | $6,789,000 |
| MDL-010 | $45,678,000 | $12,345,000 | $5,678,000 | $0 |
| MDL-011 | $23,456,000 | $0 | $0 | $0 |
| MDL-012 | $0 | $0 | $0 | $12,345,000 |

## 7. Comparison to Prior Period

### Table 7.1 — Quarter-over-Quarter Performance Changes

| Model ID | Test AUC Q4 2025 | Test AUC Q1 2026 | Delta AUC | Status Change |
|:---------|:----------------:|:----------------:|:---------:|:-------------:|
| MDL-001 | 0.8623 | 0.8412 | -0.0211 | Stable → Degraded |
| MDL-002 | 0.9312 | 0.9234 | -0.0078 | Stable → Watch |
| MDL-003 | 0.8912 | 0.8756 | -0.0156 | Watch → Degraded |
| MDL-004 | 0.8045 | 0.8012 | -0.0033 | Stable → Stable |
| MDL-005 | 0.7512 | 0.7234 | -0.0278 | Watch → Degraded |
| MDL-006 | 0.8234 | 0.8123 | -0.0111 | Stable → Watch |
| MDL-007 | 0.7401 | 0.7389 | -0.0012 | Stable → Stable |
| MDL-008 | 0.8456 | 0.8345 | -0.0111 | Stable → Watch |
| MDL-009 | 0.7556 | 0.7534 | -0.0022 | Stable → Stable |
| MDL-010 | 0.8923 | 0.8891 | -0.0032 | Stable → Stable |
| MDL-011 | 0.6612 | 0.6534 | -0.0078 | Stable → Stable |
| MDL-012 | 0.8812 | 0.8567 | -0.0245 | Stable → Degraded |

## 8. Remediation Tracking

### Table 8.1 — Open Remediation Actions

| Action ID | Finding ID | Model ID | Action | Owner | Due Date | Status | % Complete |
|:----------|:-----------|:---------|:-------|:------|:--------:|:------:|:----------:|
| ACT-001 | FND-001 | MDL-001 | Retrain with fairness constraints | Data Science Team | 2026-04-30 | In Progress | 60% |
| ACT-002 | FND-002 | MDL-001 | Add age-aware calibration layer | Data Science Team | 2026-04-30 | In Progress | 45% |
| ACT-003 | FND-003 | MDL-001 | Reduce depth, add regularization | Data Science Team | 2026-05-15 | Planned | 0% |
| ACT-004 | FND-004 | MDL-001 | Refresh bureau data pipeline | Data Engineering | 2026-04-30 | In Progress | 75% |
| ACT-005 | FND-005 | MDL-002 | Retrain on recent transaction data | Fraud Analytics | 2026-04-15 | In Progress | 80% |
| ACT-006 | FND-006 | MDL-003 | Update customer profile features | AML Team | 2026-05-01 | Planned | 10% |
| ACT-007 | FND-007 | MDL-005 | Full model rebuild with new features | Customer Analytics | 2026-04-30 | In Progress | 30% |
| ACT-008 | FND-008 | MDL-005 | Add ensemble smoothing layer | Customer Analytics | 2026-05-15 | Planned | 0% |
| ACT-009 | FND-009 | MDL-005 | Fix engagement data ETL pipeline | Data Engineering | 2026-04-15 | In Progress | 90% |
| ACT-010 | FND-010 | MDL-012 | Retrain on expanded corpus | NLP Team | 2026-05-15 | Planned | 15% |
| ACT-011 | FND-011 | MDL-012 | Add adversarial training | NLP Team | 2026-05-15 | Planned | 0% |
| ACT-012 | FND-012 | MDL-001 | Monitor bureau data completeness | Data Engineering | 2026-05-31 | In Progress | 50% |

---

*This report was prepared by the Model Validation Unit (MVU) in accordance with SR 11-7
guidance. All findings are subject to independent review by Internal Audit.*