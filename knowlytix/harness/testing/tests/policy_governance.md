# AI Governance Policy Corpus

**Authority:** Enterprise AI Risk Committee
**Effective Date:** 2026-04-01
**Version:** 1.0

This document codifies the governance policy over the agent fleet,
tool catalog, and workflow library for the enterprise AI system.
It serves as the ground-truth corpus for the policy GMS.

## 1. Agents and Roles

### Table 1.1 - Agent Catalog

| Agent ID | Role | Seniority | Authority Level |
|:---------|:-----|:---------:|:---------------:|
| model_risk_analyst | Risk Analyst | 2 | 1 |
| compliance_auditor | Auditor | 3 | 2 |
| governance_officer | Governance Officer | 5 | 4 |
| ops_engineer | Operator | 2 | 2 |
| escalation_manager | Escalation Lead | 4 | 3 |

## 2. Tools and Capabilities

### Table 2.1 - Tool Catalog

| Tool ID | Purpose | Risk Tier | Sensitivity |
|:--------|:--------|:---------:|:-----------:|
| data_lookup | Read-only query | 3 | 1 |
| risk_check | Risk scoring | 2 | 2 |
| send_notification | Send alerts | 2 | 2 |
| submit_escalation | Raise case | 1 | 3 |
| approve_action | Apply approval | 1 | 4 |
| revoke_access | Revoke credentials | 1 | 4 |
| dangerous_delete | Destructive action | 1 | 5 |

## 3. Workflows and Compliance

### Table 3.1 - Workflow Compliance Status

| Workflow | Standard | Risk Tier | Status |
|:---------|:---------|:---------:|:------:|
| sox_compliant_workflow | audit_standard_sox | 2 | pass |
| bcbs_compliant_workflow | audit_standard_bcbs | 2 | pass |
| reviewed_pipeline | governance_charter | 2 | pass |
| validated_pipeline | governance_charter | 2 | pass |
| conservative_lending_model | regulatory_threshold | 2 | pass |
| standard_risk_model | regulatory_threshold | 2 | pass |
| legacy_workflow_v1 | audit_standard_sox | 1 | fail |
| deprecated_workflow_v2 | audit_standard_bcbs | 1 | fail |
| shadow_pipeline | governance_charter | 1 | fail |
| unreviewed_deployment | governance_charter | 1 | fail |
| aggressive_growth_model | regulatory_threshold | 1 | fail |
| unaudited_workflow_model | regulatory_threshold | 1 | fail |

## 4. Standards Catalog

### Table 4.1 - Regulatory Standards

| Standard ID | Authority | Severity |
|:------------|:----------|:--------:|
| audit_standard_sox | SEC | 5 |
| audit_standard_bcbs | BIS | 5 |
| governance_charter | Internal | 4 |
| regulatory_threshold | Federal Reserve | 5 |

## 5. Rule Narrative

The enterprise AI risk committee assigns each agent a fixed tool
authorization set. Escalation actions are restricted to agents with
approval authority. Destructive tools are restricted to governance
officers only. Workflows that do not comply with at least one audit
standard are flagged for remediation.
