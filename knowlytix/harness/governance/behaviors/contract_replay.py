# SPDX-License-Identifier: Apache-2.0
"""Replay-regression battery for behavioral contracts.

A bundle records the *measured* detection rate at calibration time
(\\Cref{contract_bundle}).  But a passing rate is not the same as
deterministic, byte-identical replay of the underlying verdicts ---
two contracts could both report 100\\% detection while differing on
*which* probes they reject, or with subtly different reason
messages.  Replay regression captures exact verdicts so a future CI
run can assert ``the contract validates this cohort byte-identically
to the recorded baseline.''

The replay snapshot is a deterministic dictionary mapping each
probe's name to a hash of its validation result.  Replay-equality is
hash-equality on every probe.  Use:

* :func:`replay_contract` to compute a snapshot from a contract +
  cohort pair.
* :func:`assert_no_contract_regression` to compare a candidate
  snapshot against a baseline; raises with the specific divergent
  probes named.

The pattern mirrors :mod:`replay` (Ch.10) and :mod:`policy_replay`
(Ch.12); the same shape across runtimes lets operators reason about
all three regression batteries the same way.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from knowlytix.harness.governance.behaviors.contracts import (
    BehavioralContract,
    ProcessEvaluator,
)


@dataclass
class ContractReplayResult:
    """Per-probe validation snapshot.

    ``per_probe`` maps the probe name to a SHA-256 hash of its
    canonicalized validation outcome (valid bool, ordered violation
    list, ordered states-visited list).  The hash is robust to
    ordering of unrelated metadata but sensitive to any change in
    *what was reported*, so a contract that rejects the same probes
    but for different reasons produces a different hash.

    ``contract_fingerprint`` and ``cohort_hash`` are stamped on the
    snapshot so a comparison can verify the two snapshots came from
    the same contract+cohort pair (otherwise differences are
    expected and the comparison is meaningless).
    """

    contract_name: str
    contract_fingerprint: str
    cohort_hash: str
    per_probe: dict[str, str]


class ContractReplayRegressionError(AssertionError):
    """Raised when replay diverges from the baseline."""


def _hash_validation(
    valid: bool,
    violations: list[str],
    states_visited: list[str],
) -> str:
    h = hashlib.sha256()
    payload = json.dumps(
        {
            "valid": valid,
            "violations": list(violations),
            "states_visited": list(states_visited),
        },
        sort_keys=True,
    )
    h.update(payload.encode("utf-8"))
    return h.hexdigest()


def replay_contract(
    contract: BehavioralContract,
    cohort,
    *,
    evaluator: ProcessEvaluator | None = None,
) -> ContractReplayResult:
    """Run every probe through the contract and snapshot the verdicts.

    The result is stable across processes and host environments
    because validation is pure-Python over hashable inputs --- no
    floating-point, no ordering hazards.
    """
    from knowlytix.harness.governance.behaviors.contract_adversarial import cohort_hash
    from knowlytix.harness.governance.behaviors.contract_bundle import (
        compute_contract_fingerprint,
    )

    if evaluator is None:
        evaluator = ProcessEvaluator(contract)

    per_probe: dict[str, str] = {}
    for probe in cohort:
        result = evaluator.validate_sequence(
            list(probe.sequence), contract,
        )
        per_probe[probe.name] = _hash_validation(
            valid=result.valid,
            violations=result.violations,
            states_visited=[s.value for s in result.states_visited],
        )

    fingerprint = compute_contract_fingerprint(contract)
    fp_hash = hashlib.sha256(
        json.dumps(
            {
                "states": fingerprint.states_hash,
                "transitions": fingerprint.transitions_hash,
                "permissions": fingerprint.permissions_hash,
                "invariants": fingerprint.invariants_hash,
                "budget": fingerprint.budget,
                "human_approval": fingerprint.require_human_approval,
            },
            sort_keys=True,
        ).encode("utf-8"),
    ).hexdigest()

    return ContractReplayResult(
        contract_name=contract.name,
        contract_fingerprint=fp_hash,
        cohort_hash=cohort_hash(cohort),
        per_probe=per_probe,
    )


def assert_no_contract_regression(
    baseline: ContractReplayResult,
    candidate: ContractReplayResult,
) -> None:
    """Raise unless the candidate snapshot matches the baseline byte-for-byte.

    Comparison checks (in order):

    1. Same contract name (mismatched names indicate the snapshots
       were taken against different contracts and a comparison is
       meaningless).
    2. Same contract fingerprint (the contract has not been
       modified between baseline and candidate).
    3. Same cohort hash (the cohort has not been edited).
    4. Per-probe hashes match for every probe.
    """
    if baseline.contract_name != candidate.contract_name:
        raise ContractReplayRegressionError(
            f"Contract names differ: baseline={baseline.contract_name!r}, "
            f"candidate={candidate.contract_name!r}"
        )
    if baseline.contract_fingerprint != candidate.contract_fingerprint:
        raise ContractReplayRegressionError(
            "Contract fingerprint changed --- contract has been modified "
            "since baseline.  Recompute the baseline or revert the "
            "contract change."
        )
    if baseline.cohort_hash != candidate.cohort_hash:
        raise ContractReplayRegressionError(
            "Cohort hash changed --- the adversarial cohort has been "
            "edited since baseline.  Recompute the baseline."
        )

    diffs: list[tuple[str, str, str]] = []
    keys = set(baseline.per_probe) | set(candidate.per_probe)
    for k in sorted(keys):
        b = baseline.per_probe.get(k)
        c = candidate.per_probe.get(k)
        if b != c:
            diffs.append((k, b or "<missing>", c or "<missing>"))

    if diffs:
        head = "\n  ".join(
            f"{name}: baseline={b[:12]} candidate={c[:12]}"
            for name, b, c in diffs[:10]
        )
        raise ContractReplayRegressionError(
            f"{len(diffs)} probes diverge from baseline:\n  {head}"
            + ("\n  ..." if len(diffs) > 10 else "")
        )
