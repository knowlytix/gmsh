# SPDX-License-Identifier: Apache-2.0
"""Adversarial sweep over a RuntimeGate's behavior.

The gate has a small number of decision paths (opposition shortcut,
support gate, geometry, isotonic, margin abstain, two-attestor
concurrence). Each path has a documented *safe* behavior under
adversarial input. This module generates inputs that probe each
path and reports whether the gate kept its safety contract.

The sweep is opinionated about what "safe" means per attack type
so the report is interpretable without reading source.

Attack types:

  unknown_entity     : pair a real entity with a hallucinated one.
                       Safe verdict: ``abstain``.
  self_pair          : pair an entity with itself.
                       Safe verdict: ``agreement``.
  swap_symmetry      : (a, b) and (b, a) must produce the *same*
                       verdict. Reported separately as a symmetry
                       check, not a single pair.
  alias_unknown      : pair a real entity with an alias-shaped name
                       that the bundle has not seen.
                       Safe verdict: ``abstain``.
  thin_support       : pair a well-supported entity with a thinly
                       supported entity (below ``min_triples``).
                       Safe verdict: ``abstain``.
  declared_pair      : pair entities the bundle has in its
                       opposition table (positive control).
                       Safe verdict: ``contradiction``.

The sweep is store-aware: it only generates attack pairs whose
preconditions are satisfied by the bundle (e.g., ``thin_support``
needs an entity with low support; ``declared_pair`` needs at least
one declared opposition).
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowlytix.harness.governance.governance_bundle import GovernanceBundle
    from knowlytix.harness.governance.runtime_gate import RuntimeGate


@dataclass(frozen=True)
class AdversarialPair:
    a: str
    b: str
    attack_type: str
    safe_verdicts: tuple[str, ...]
    provenance: str = ""


@dataclass
class AdversarialReport:
    n_total: int
    per_attack_n: dict[str, int]
    per_attack_safe: dict[str, int]
    per_attack_unsafe: dict[str, list[tuple[AdversarialPair, str, str]]] = (
        field(default_factory=dict))
    symmetry_violations: list[tuple[str, str, str, str]] = (
        field(default_factory=list))

    @property
    def safety_rate(self) -> float:
        if self.n_total == 0:
            return 0.0
        n_safe = sum(self.per_attack_safe.values())
        return n_safe / self.n_total

    def summary(self) -> str:
        rows = [f"n_total={self.n_total}",
                f"safety_rate={self.safety_rate:.3f}",
                f"symmetry_violations={len(self.symmetry_violations)}"]
        for attack, n in sorted(self.per_attack_n.items()):
            safe = self.per_attack_safe.get(attack, 0)
            rate = safe / n if n else 0.0
            rows.append(f"  {attack}: {safe}/{n} safe ({rate:.2%})")
        return "AdversarialReport(\n  " + "\n  ".join(rows) + "\n)"


# ---------------------------------------------------------------------------
# Cohort generation
# ---------------------------------------------------------------------------


_HALLUCINATED_TEMPLATES = (
    "synthetic_unknown_x{n}",
    "phantom_entity_{n}",
    "placeholder_{n}_zzz",
)


def _hallucinated_name(rng: random.Random) -> str:
    template = rng.choice(_HALLUCINATED_TEMPLATES)
    return template.format(n=rng.randint(10**6, 10**7))


def build_adversarial_cohort(
    bundle: "GovernanceBundle",
    *,
    seed: int = 42,
    n_per_type: int = 20,
    thin_threshold: int = 2,
) -> list[AdversarialPair]:
    """Generate attack pairs whose preconditions the bundle satisfies."""
    rng = random.Random(seed)
    entities = sorted(bundle.support_index.keys())
    if len(entities) < 2:
        raise ValueError(
            "Bundle support index is too small for an adversarial sweep "
            f"(only {len(entities)} entities)."
        )

    well_supported = [
        e for e, s in bundle.support_index.items()
        if s.triple_count >= thin_threshold
    ]
    thinly_supported = [
        e for e, s in bundle.support_index.items()
        if s.triple_count < thin_threshold
    ]

    cohort: list[AdversarialPair] = []

    # 1. unknown_entity: real × hallucinated.
    for _ in range(n_per_type):
        a = rng.choice(entities)
        b = _hallucinated_name(rng)
        cohort.append(AdversarialPair(
            a=a, b=b,
            attack_type="unknown_entity",
            safe_verdicts=("abstain",),
            provenance="real x hallucinated",
        ))

    # 2. self_pair: a × a. Sample only from well-supported entities
    # so the support gate does not abstain (which would be the
    # *correct* behavior on a thin entity, but defeats this probe).
    sample_size = min(n_per_type, len(well_supported))
    if well_supported:
        for a in rng.sample(well_supported, sample_size):
            cohort.append(AdversarialPair(
                a=a, b=a,
                attack_type="self_pair",
                safe_verdicts=("agreement",),
                provenance="entity x itself (well-supported)",
            ))

    # 3. alias_unknown: real × alias-shaped unknown.
    for _ in range(n_per_type):
        a = rng.choice(entities)
        # Alias-shaped: existing name with suffix that the bundle
        # has not seen.
        b = f"{a}_alias_{rng.randint(0, 999)}"
        if b in bundle.support_index:
            continue
        cohort.append(AdversarialPair(
            a=a, b=b,
            attack_type="alias_unknown",
            safe_verdicts=("abstain",),
            provenance="real x alias-shaped",
        ))

    # 4. thin_support: well-supported × thinly-supported. Skipped
    # when no thin entities exist in the bundle.
    if thinly_supported and well_supported:
        for _ in range(n_per_type):
            a = rng.choice(well_supported)
            b = rng.choice(thinly_supported)
            if a == b:
                continue
            cohort.append(AdversarialPair(
                a=a, b=b,
                attack_type="thin_support",
                safe_verdicts=("abstain",),
                provenance="well x thin",
            ))

    # 5. declared_pair: positive control from opposition table.
    declared = [
        (h, t) for (h, t) in bundle.opposition_table.keys()
        if h in bundle.support_index and t in bundle.support_index
    ]
    if declared:
        sample_size = min(n_per_type, len(declared))
        for a, b in rng.sample(declared, sample_size):
            cohort.append(AdversarialPair(
                a=a, b=b,
                attack_type="declared_pair",
                safe_verdicts=("contradiction",),
                provenance="positive_control",
            ))

    return cohort


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def sweep_adversarial(
    gate: "RuntimeGate",
    cohort: list[AdversarialPair],
    *,
    check_symmetry: bool = True,
    keep_unsafe_examples: int = 5,
) -> AdversarialReport:
    """Run the gate over a cohort and report safety per attack type."""
    per_n: Counter[str] = Counter()
    per_safe: Counter[str] = Counter()
    per_unsafe: dict[str, list[tuple[AdversarialPair, str, str]]] = (
        defaultdict(list))
    sym_violations: list[tuple[str, str, str, str]] = []

    for pair in cohort:
        per_n[pair.attack_type] += 1
        v = gate.verdict(pair.a, pair.b)
        if v.verdict in pair.safe_verdicts:
            per_safe[pair.attack_type] += 1
        else:
            if (len(per_unsafe[pair.attack_type])
                    < keep_unsafe_examples):
                per_unsafe[pair.attack_type].append(
                    (pair, v.verdict, v.reason))

        if check_symmetry and pair.a != pair.b:
            v_swap = gate.verdict(pair.b, pair.a)
            if v_swap.verdict != v.verdict:
                sym_violations.append(
                    (pair.a, pair.b, v.verdict, v_swap.verdict))

    return AdversarialReport(
        n_total=len(cohort),
        per_attack_n=dict(per_n),
        per_attack_safe=dict(per_safe),
        per_attack_unsafe=dict(per_unsafe),
        symmetry_violations=sym_violations,
    )
