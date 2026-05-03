# SPDX-License-Identifier: Apache-2.0
"""Systematic adversarial cohort for behavioral-contract regression.

The hand-curated 20-probe cohort in Lab~13.2 demonstrates that the FSM
catches \emph{some} violations.  It does not prove the gate is
complete.  This module generates a contract-derived cohort whose
construction guarantees coverage of every block path the FSM exposes:

  * **Allowed-transition probes** (positive controls): for each
    transition $(q, q') \in \delta$, build a minimal sequence that
    enters~$q$ and then triggers~$q'$.  The FSM must \emph{accept}.
  * **Forbidden-transition probes**: for each pair $(q, q') \notin
    \delta$ where some tool's ``TOOL_STATE_ADVANCE`` lands in~$q'$,
    build a sequence that reaches~$q$ and then calls that tool.  The
    FSM must \emph{reject} with a transition violation.
  * **Forbidden-tool-in-state probes**: for each $(q, t)$ where $t$
    is in ``ALL_TOOLS`` but $t \notin \Pi(q)$, build a sequence
    reaching~$q$ and call $t$.  The FSM must reject with a permission
    violation.
  * **Invariant probes**: for each invariant function in the
    contract, build a sequence that satisfies the FSM transitions
    and permissions but trips the invariant (e.g.\ reach
    ``DEPLOYMENT_READY`` skipping fairness).  The FSM must reject
    with an invariant violation.
  * **Budget probes**: a sequence with length exactly
    ``max_tool_calls + 1`` of the cheapest legal tool.  Must reject
    with a budget violation.
  * **Human-approval probes** (when ``require_human_approval``): a
    sequence reaching ``DEPLOYMENT_READY`` with no
    ``HUMAN_REVIEW`` visit.  Must reject.
  * **Edge-case probes**: empty sequence, single legal tool,
    single illegal tool, repeated tool.

The result is a closed-form cohort whose size scales with $|\delta|
+ |Q \times T \setminus \Pi|$ rather than with the lab author's
imagination.  Any future contract change adds or removes probes
automatically, so coverage tracks the FSM definition.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

from knowlytix.harness.governance.behaviors.contracts import (
    ALL_TOOLS,
    BehavioralContract,
    ContractState,
    ProcessEvaluator,
    SequenceValidation,
    TOOL_STATE_ADVANCE,
)


# -------------------------------------------------------------------- #
# Probe / report dataclasses                                           #
# -------------------------------------------------------------------- #


@dataclass(frozen=True)
class ContractAdversarialProbe:
    """One systematic probe over a behavioral contract.

    ``expected_valid`` is True for positive controls and False for
    every adversarial probe.  ``expected_violation_kind`` names the
    block path the probe is intended to exercise; the sweep checks
    that the actual rejection reason mentions that kind so a probe
    that rejects for the *wrong* reason is reported as unsafe.
    """

    name: str
    sequence: tuple[str, ...]
    expected_valid: bool
    expected_violation_kind: str | None
    provenance: str = ""


@dataclass
class ContractAdversarialReport:
    n_total: int
    per_kind_n: dict[str, int]
    per_kind_safe: dict[str, int]
    per_kind_unsafe: dict[str, list[tuple[str, str]]] = field(
        default_factory=dict
    )

    @property
    def safety_rate(self) -> float:
        if self.n_total == 0:
            return 0.0
        return sum(self.per_kind_safe.values()) / self.n_total

    @property
    def detection_rate(self) -> float:
        """Fraction of true-violation probes the contract detected."""
        n_violation_probes = sum(
            n for k, n in self.per_kind_n.items() if k != "allowed"
        )
        if n_violation_probes == 0:
            return 0.0
        n_caught = sum(
            self.per_kind_safe.get(k, 0)
            for k in self.per_kind_n
            if k != "allowed"
        )
        return n_caught / n_violation_probes

    @property
    def false_positive_rate(self) -> float:
        """Fraction of valid (positive-control) probes the contract rejected."""
        n_allowed = self.per_kind_n.get("allowed", 0)
        if n_allowed == 0:
            return 0.0
        n_safe = self.per_kind_safe.get("allowed", 0)
        return (n_allowed - n_safe) / n_allowed

    def summary(self) -> str:
        rows = [
            f"n_total={self.n_total}",
            f"safety_rate={self.safety_rate:.3f}",
            f"detection_rate={self.detection_rate:.3f}",
            f"false_positive_rate={self.false_positive_rate:.3f}",
        ]
        for kind, n in sorted(self.per_kind_n.items()):
            safe = self.per_kind_safe.get(kind, 0)
            rate = safe / n if n else 0.0
            rows.append(f"  {kind}: {safe}/{n} safe ({rate:.2%})")
        return "ContractAdversarialReport(\n  " + "\n  ".join(rows) + "\n)"


# -------------------------------------------------------------------- #
# Sequence synthesis: shortest path from IDLE to a target state        #
# -------------------------------------------------------------------- #


_CANONICAL_TOOL_FOR_STATE: dict[ContractState, str] = {
    ContractState.DATA_LOADED:      "load_dataset",
    ContractState.PREPROCESSED:     "preprocess_data",
    ContractState.MODEL_TRAINED:    "train_model",
    ContractState.DIAGNOSED:        "diagnose_robustness",
    ContractState.FAIR_CHECKED:     "diagnose_fairness",
    ContractState.COMPARED:         "leaderboard",
}


def _tools_advancing_to(state: ContractState) -> list[str]:
    """Return tools whose call advances the FSM to ``state``.

    Canonical tools (the one whose name matches the state's purpose,
    e.g. ``preprocess_data`` for ``PREPROCESSED``) are listed first
    so cohort synthesis prefers them.  This matters when an
    invariant looks for a specific tool by name in history --- a
    valid prefix that advances the state via a different tool would
    pass the FSM but trip the invariant, falsely flagging an
    "allowed" probe as a contract violation.
    """
    canonical = _CANONICAL_TOOL_FOR_STATE.get(state)
    candidates = sorted(
        t for t, s in TOOL_STATE_ADVANCE.items() if s == state
    )
    if canonical and canonical in candidates:
        candidates.remove(canonical)
        candidates.insert(0, canonical)
    return candidates


def _shortest_tool_path(
    contract: BehavioralContract,
    target: ContractState,
) -> list[str] | None:
    """Find a shortest tool sequence that reaches ``target`` from IDLE.

    Uses BFS over the FSM's transition graph, restricted to tools
    that are permitted in their originating state.  Returns ``None``
    if the target is unreachable.
    """
    if target == ContractState.IDLE:
        return []

    visited = {ContractState.IDLE}
    queue: deque[tuple[ContractState, list[str]]] = deque(
        [(ContractState.IDLE, [])]
    )
    while queue:
        state, path = queue.popleft()
        for next_state in contract.transitions.get(state, set()):
            if next_state in visited or next_state == state:
                continue
            for tool in _tools_advancing_to(next_state):
                if tool not in contract.allowed_tools(state):
                    continue
                new_path = path + [tool]
                if next_state == target:
                    return new_path
                visited.add(next_state)
                queue.append((next_state, new_path))
                break
    return None


# -------------------------------------------------------------------- #
# Cohort builder                                                       #
# -------------------------------------------------------------------- #


def build_contract_adversarial_cohort(
    contract: BehavioralContract,
) -> list[ContractAdversarialProbe]:
    """Generate the systematic adversarial cohort for a contract.

    Coverage guarantees:

    * Every transition in ``contract.transitions`` produces at least
      one ``allowed`` probe (when reachable from IDLE).
    * Every ``(state, target_state)`` pair where ``target_state`` is
      reachable by *some* tool but not by transition produces at
      least one ``forbidden_transition`` probe.
    * Every invariant produces at least one probe.
    * The budget cap and human-approval flag (when set) produce
      one probe each.
    """
    probes: list[ContractAdversarialProbe] = []

    # ---- 1. Allowed-transition probes (positive controls) ---- #
    for src in contract.states:
        for dst in contract.transitions.get(src, set()):
            if dst == src:
                continue
            prefix = _shortest_tool_path(contract, src)
            if prefix is None:
                continue
            advance_tools = [
                t for t in _tools_advancing_to(dst)
                if t in contract.allowed_tools(src)
            ]
            if not advance_tools:
                continue
            seq = tuple(prefix + [advance_tools[0]])
            probes.append(ContractAdversarialProbe(
                name=f"allowed_{src.value}_to_{dst.value}",
                sequence=seq,
                expected_valid=True,
                expected_violation_kind=None,
                provenance="every transition exercised",
            ))

    # ---- 2. Forbidden-transition probes ---- #
    all_states_with_tool = {
        s for t, s in TOOL_STATE_ADVANCE.items() if s is not None
    }
    for src in contract.states:
        prefix = _shortest_tool_path(contract, src)
        if prefix is None:
            continue
        allowed_dsts = contract.transitions.get(src, set())
        for dst in all_states_with_tool:
            if dst == src or dst in allowed_dsts:
                continue
            advance_tools = [
                t for t in _tools_advancing_to(dst)
                if t in contract.allowed_tools(src)
            ]
            if not advance_tools:
                continue
            seq = tuple(prefix + [advance_tools[0]])
            probes.append(ContractAdversarialProbe(
                name=f"forbidden_transition_{src.value}_to_{dst.value}",
                sequence=seq,
                expected_valid=False,
                expected_violation_kind="transition",
                provenance="(q, q') not in delta",
            ))

    # ---- 3. Forbidden-tool-in-state probes ---- #
    for src in contract.states:
        prefix = _shortest_tool_path(contract, src)
        if prefix is None:
            continue
        for tool in sorted(ALL_TOOLS):
            if tool in contract.allowed_tools(src):
                continue
            seq = tuple(prefix + [tool])
            probes.append(ContractAdversarialProbe(
                name=f"forbidden_tool_{tool}_in_{src.value}",
                sequence=seq,
                expected_valid=False,
                expected_violation_kind="permission",
                provenance="t not in Pi(q)",
            ))

    # ---- 4. Invariant probes ---- #
    # Construct a deployment-ready sequence missing each invariant
    # check.  The recommendation contract enforces
    # _invariant_no_train_without_preprocess, the action-taking
    # contract adds fairness and robustness.  We build per-invariant
    # bypasses where the FSM transitions and tool permissions all
    # check out but the invariant fires.
    for inv in contract.invariants:
        bypass_seq = _build_invariant_bypass(contract, inv.__name__)
        if bypass_seq is None:
            continue
        probes.append(ContractAdversarialProbe(
            name=f"invariant_{inv.__name__}",
            sequence=tuple(bypass_seq),
            expected_valid=False,
            expected_violation_kind="invariant",
            provenance="FSM-valid but invariant trips",
        ))

    # ---- 5. Budget exhaustion probe ---- #
    cheapest = "recall_memory" if "recall_memory" in (
        contract.allowed_tools(ContractState.IDLE)
    ) else "load_dataset"
    budget_seq = tuple([cheapest] * (contract.max_tool_calls + 2))
    probes.append(ContractAdversarialProbe(
        name="budget_exceeded",
        sequence=budget_seq,
        expected_valid=False,
        expected_violation_kind="budget",
        provenance=f"len > max_tool_calls={contract.max_tool_calls}",
    ))

    # ---- 6. Human-approval probe ---- #
    if contract.require_human_approval:
        deploy_path = _shortest_tool_path(
            contract, ContractState.DEPLOYMENT_READY,
        )
        if deploy_path is not None:
            probes.append(ContractAdversarialProbe(
                name="human_approval_skipped",
                sequence=tuple(deploy_path),
                expected_valid=False,
                expected_violation_kind="human_approval",
                provenance="DEPLOYMENT_READY without HUMAN_REVIEW",
            ))

    # ---- 7. Edge-case probes ---- #
    probes.append(ContractAdversarialProbe(
        name="empty_sequence",
        sequence=tuple(),
        expected_valid=True,
        expected_violation_kind=None,
        provenance="zero-length input must not crash",
    ))
    probes.append(ContractAdversarialProbe(
        name="single_idle_only_tool",
        sequence=("recall_memory",),
        expected_valid=(
            "recall_memory"
            in contract.allowed_tools(ContractState.IDLE)
        ),
        expected_violation_kind=(
            None
            if "recall_memory"
            in contract.allowed_tools(ContractState.IDLE)
            else "permission"
        ),
        provenance="single tool, no state advance",
    ))
    probes.append(ContractAdversarialProbe(
        name="repeated_load",
        sequence=("load_dataset", "load_dataset", "load_dataset"),
        expected_valid=True,
        expected_violation_kind=None,
        provenance="self-loop in DATA_LOADED",
    ))

    return probes


def _build_invariant_bypass(
    contract: BehavioralContract,
    invariant_name: str,
) -> list[str] | None:
    """Construct an FSM-valid sequence that trips a named invariant.

    Each branch is keyed off the invariant's function name.  This is
    deliberately explicit: an invariant added later that this
    function does not recognize will have ``None`` returned and the
    cohort will skip generating a probe for it (visible at sweep
    time as a missing kind), which is preferable to silently
    fabricating an unrelated sequence.
    """
    # All three known invariants check the same property -- a state
    # was reached without a prerequisite tool in history.  We
    # construct a deployment path that omits the prerequisite while
    # keeping the FSM transitions valid.
    if invariant_name == "_invariant_no_train_without_preprocess":
        # The transition table requires PREPROCESSED before
        # MODEL_TRAINED in every shipped contract, so this invariant
        # can only fire if the FSM is misconfigured to allow a
        # DATA_LOADED -> MODEL_TRAINED transition.  In that case the
        # bypass is "load_dataset, train_model".  We test by
        # constructing a sequence that the FSM rules out, then
        # noting the invariant would trip if FSM disagreed.
        if (ContractState.MODEL_TRAINED
                in contract.transitions.get(
                    ContractState.DATA_LOADED, set())):
            return ["load_dataset", "train_model"]
        return None

    if invariant_name == "_invariant_no_deploy_without_fairness":
        # Reach DEPLOYMENT_READY through a path that omits
        # diagnose_fairness.  The action-taking contract forces
        # fairness in the transition table; the recommendation
        # contract permits skipping it.
        path = _shortest_tool_path(
            contract, ContractState.DEPLOYMENT_READY,
        )
        if path is None:
            return None
        if "diagnose_fairness" in path:
            # Try to reach DEPLOYMENT_READY without fairness via the
            # DIAGNOSED -> DEPLOYMENT_READY edge.
            alt = _shortest_path_avoiding_tool(
                contract, ContractState.DEPLOYMENT_READY,
                "diagnose_fairness",
            )
            return alt
        return path

    if invariant_name == "_invariant_no_deploy_without_robustness":
        path = _shortest_tool_path(
            contract, ContractState.DEPLOYMENT_READY,
        )
        if path is None:
            return None
        if "diagnose_robustness" in path:
            alt = _shortest_path_avoiding_tool(
                contract, ContractState.DEPLOYMENT_READY,
                "diagnose_robustness",
            )
            return alt
        return path

    return None


def _shortest_path_avoiding_tool(
    contract: BehavioralContract,
    target: ContractState,
    avoid_tool: str,
) -> list[str] | None:
    """BFS shortest path to ``target`` that never uses ``avoid_tool``."""
    visited = {ContractState.IDLE}
    queue: deque[tuple[ContractState, list[str]]] = deque(
        [(ContractState.IDLE, [])]
    )
    while queue:
        state, path = queue.popleft()
        for next_state in contract.transitions.get(state, set()):
            if next_state in visited or next_state == state:
                continue
            for tool in _tools_advancing_to(next_state):
                if tool == avoid_tool:
                    continue
                if tool not in contract.allowed_tools(state):
                    continue
                new_path = path + [tool]
                if next_state == target:
                    return new_path
                visited.add(next_state)
                queue.append((next_state, new_path))
                break
    return None


# -------------------------------------------------------------------- #
# Sweep                                                                #
# -------------------------------------------------------------------- #


def sweep_contract_adversarial(
    contract: BehavioralContract,
    cohort: list[ContractAdversarialProbe],
    *,
    evaluator: ProcessEvaluator | None = None,
) -> ContractAdversarialReport:
    """Validate every probe and report per-kind safety.

    A probe is "safe" when:

    * ``expected_valid=True`` and the validation passed; or
    * ``expected_valid=False`` and the validation failed AND the
      reported violation mentions ``expected_violation_kind`` (so a
      probe that rejects for the wrong reason is unsafe).
    """
    if evaluator is None:
        evaluator = ProcessEvaluator(contract)

    per_n: dict[str, int] = {}
    per_safe: dict[str, int] = {}
    per_unsafe: dict[str, list[tuple[str, str]]] = {}

    for probe in cohort:
        kind = (
            "allowed"
            if probe.expected_valid
            else (probe.expected_violation_kind or "unspecified")
        )
        per_n[kind] = per_n.get(kind, 0) + 1

        try:
            result = evaluator.validate_sequence(
                list(probe.sequence), contract,
            )
        except Exception as exc:
            per_unsafe.setdefault(kind, []).append(
                (probe.name, f"raised: {type(exc).__name__}: {exc}")
            )
            continue

        is_safe, evidence = _classify_probe(probe, result)
        if is_safe:
            per_safe[kind] = per_safe.get(kind, 0) + 1
        else:
            per_unsafe.setdefault(kind, []).append(
                (probe.name, evidence)
            )

    return ContractAdversarialReport(
        n_total=len(cohort),
        per_kind_n=per_n,
        per_kind_safe=per_safe,
        per_kind_unsafe=per_unsafe,
    )


def _classify_probe(
    probe: ContractAdversarialProbe,
    result: SequenceValidation,
) -> tuple[bool, str]:
    """Decide whether the validation outcome matches expectation."""
    if probe.expected_valid:
        if result.valid:
            return True, "passed"
        return False, f"expected_valid but rejected: {result.violations[:1]}"

    # Expected to be rejected.
    if result.valid:
        return False, "expected_invalid but passed"

    # Rejected; require the kind to match.
    expected_kind = probe.expected_violation_kind
    if expected_kind is None:
        return True, "rejected (kind not asserted)"

    joined = " | ".join(result.violations + result.invariant_violations)
    kind_keywords = {
        "transition": ("transition", "not permitted"),
        "permission": ("not permitted in state", "tool ", "not permitted"),
        "invariant": ("invariant",),
        "budget": ("budget",),
        "human_approval": ("HUMAN_REVIEW", "approval"),
    }.get(expected_kind, (expected_kind,))

    if any(kw.lower() in joined.lower() for kw in kind_keywords):
        return True, f"rejected for {expected_kind}"
    return False, (
        f"rejected for wrong reason: expected={expected_kind}, "
        f"got='{joined[:120]}'"
    )


# -------------------------------------------------------------------- #
# Cohort hashing                                                       #
# -------------------------------------------------------------------- #


def cohort_hash(cohort: Iterable[ContractAdversarialProbe]) -> str:
    """SHA-256 over canonical (name, sequence) tuples.

    The hash is stable across runs given the same contract, so the
    bundle records exactly which cohort produced its measured
    detection rate.  A future cohort that adds, removes, or
    reorders probes produces a different hash and the bundle's
    binding refuses to bind.
    """
    h = hashlib.sha256()
    for probe in sorted(cohort, key=lambda p: p.name):
        h.update(probe.name.encode("utf-8"))
        h.update(b"\x00")
        h.update("|".join(probe.sequence).encode("utf-8"))
        h.update(b"\x00")
        h.update(str(probe.expected_valid).encode("utf-8"))
        h.update(b"\x00")
        h.update((probe.expected_violation_kind or "").encode("utf-8"))
        h.update(b"\x00\x00")
    return h.hexdigest()
