# Policy Bundle Regression Log

This log tracks every blessed-hash change for the policy bundle
(`expected_policy_replay_hash.txt`). Every entry must include:
the hash, the date it was blessed, who approved the change, and
the reason. The CI ship gate refuses to ship a bundle whose
replay hash does not match the latest blessed value.

## Policy Bundle (Ch 12)

| Hash               | Date       | Approver                   | Cohort     | Reason                                      |
| ------------------ | ---------- | -------------------------- | ---------- | ------------------------------------------- |
| `7359d07153302322` | 2026-04-27 | initial-bless              | v1 (30)    | Lab 12.4 first-bless on 30-probe cohort     |
| `d0c019adbb411daa` | 2026-04-27 | initial-bless              | v2 (1133)  | Lab 12.5 hardening: cohort scaled to 1133, alt-validator + multi-signer + FSM invariants + latency benchmark + drift simulation added |

## Audit Trail Bundle (Ch 14)

Tracks the metrics-fingerprint anchor for the audit-trail
hardening pipeline (file: `expected_audit_metrics_fingerprint.txt`).

| Fingerprint        | Date       | Approver                   | Cohort     | Reason                                      |
| ------------------ | ---------- | -------------------------- | ---------- | ------------------------------------------- |
| `601a6037bd6bddce` | 2026-04-27 | initial-bless              | v2 (60)    | Lab 14.4 first-bless: 60-probe v2 cohort, FSM invariants 29/29, alt-validator 35/35, multi-signer + latency benchmarks |

## Workflow Bundle (Ch 16)

Tracks the snapshot fingerprint for the integrated Adult Income
workflow (file: `expected_workflow_fingerprint.txt`).

| Fingerprint        | Date       | Approver                   | Workflow         | Reason                                      |
| ------------------ | ---------- | -------------------------- | ---------------- | ------------------------------------------- |
| `014ab302607808bb` | 2026-04-27 | initial-bless              | Adult Income     | Lab 16.4 first-bless: 16 tool calls, 142 claims, pass_rate 0.7958 (BLOCKED), v2 cohort smoke 30/30, FSM 39/39, alt-validator 1/1, multi-signer + latency p99 20μs |

## Attack Bundle (Ch 18)

Tracks the snapshot fingerprint for the Ch.18 adversarial-suite
canonical run (file: `expected_attack_fingerprint.txt`).

| Fingerprint        | Date       | Approver                   | Suite            | Reason                                      |
| ------------------ | ---------- | -------------------------- | ---------------- | ------------------------------------------- |
| `028241f417184ec2` | 2026-04-27 | initial-bless              | v1 (5 attacks)   | Lab 18.4 first-bless: 5/5 blocked, cross-layer 5/5, v2 cohort smoke 35/35, FSM 38/38, alt-validator 5/5, multi-signer + latency p99 9μs |

## Calibration Bundle (Ch 19)

Tracks the snapshot fingerprint for the Ch.19 calibrated
threshold vector (file: `expected_calibration_fingerprint.txt`).

| Fingerprint        | Date       | Approver                   | Domain      | Reason                                      |
| ------------------ | ---------- | -------------------------- | ----------- | ------------------------------------------- |
| `c4f1c642387d5590` | 2026-04-27 | initial-bless              | iris        | Lab 19.4 first-bless (synthetic 9-config sweep): optimum theta_sim=0.5 at acc=0.95, FSM 19/19, alt-validator 2/2, cross-domain drift report Iris vs California, multi-signer + latency p99 30μs |

## Release Gate Bundle (Ch 15)

Tracks the replay-battery hash for the tiered release-gate
bundle (file: `expected_release_gate_replay_hash.txt`).

| Hash               | Date       | Approver                   | Cohort        | Reason                                      |
| ------------------ | ---------- | -------------------------- | ------------- | ------------------------------------------- |
| `e9d5e8c19625e301` | 2026-04-27 | lab15_runbook              | v1 (1200)     | Lab 15.4 first-bless: 1200-probe Adult-Income cohort, calibrated thresholds (action_taking acc=0.892 conf=0.708 rob=0.010 air=0.906), monotonicity 14641/14641, adversarial 200/200, alt-validator 690/690, latency p99=7us |

## Multi-Agent Bundle (Ch 17)

Tracks the replay-battery hash for the multi-agent governance
bundle (file: `expected_multi_agent_replay_hash.txt`).

| Hash               | Date       | Approver                   | Cohort        | Reason                                      |
| ------------------ | ---------- | -------------------------- | ------------- | ------------------------------------------- |
| `ec4d991b7b1c877f` | 2026-04-27 | lab17_runbook              | v1 (45+1080)  | Lab 17.4 first-bless: 4-role Adult Income team (DataAgent, TrainingAgent, DiagnosticsAgent, FairnessAgent); invariants 5/5 (90 cells, 0 violations); adversarial 200/200 across 10 classes (role_hijack, frozen_role_mutation, tool_alias drove real library hardening); alt-validator 76/76; naive baseline uplift +720; latency can_execute p99=0.30us |

## Approval workflow

1. Author opens a PR with the proposed change (rule YAML edit,
   store retrain, calibration adjustment, etc.).
2. CI runs `scripts/gate_policy_bundle.py` against the existing
   blessed hash. If verdicts changed, the gate fails.
3. Author rebuilds the bundle locally, captures the new hash,
   updates `expected_policy_replay_hash.txt` and this log in
   the same PR.
4. PR is reviewed by an approver outside the author's chain
   (MRM lead or compliance officer). The approver signs off on
   the verdict diff: every probe whose decision flipped must
   have a documented rationale.
5. After merge, CI re-runs the gate against the new blessed
   hash and the bundle is promoted to production.

## Rollback

If a deployed bundle is found to be broken (signing key
compromised, replay regression caught post-ship), revert this
file to the prior commit and redeploy the previous bundle
artefact. The deployment registry retains every shipped bundle
for at least 90 days.

## Compliance Bundle (Ch 20)

Tracks the snapshot fingerprint for the Ch.20 compliance
evidence package (file: `expected_compliance_fingerprint.txt`).

| Fingerprint        | Date       | Approver                   | Suite       | Reason                                      |
| ------------------ | ---------- | -------------------------- | ----------- | ------------------------------------------- |
| `3ab32d76c042b802` | 2026-04-27 | initial-bless              | runbook v1  | Lab 20.4 first-bless: PARTIAL status (5/9 satisfied), SR 11-7 0.67, EU AI Act 0.33; FSM 19/19, alt 1/1, multi-signer + latency p99 20μs; runbook uses synthetic pipeline_results, notebook produces COMPLIANT 9/9 with fingerprint dac408bb36316a4e |
