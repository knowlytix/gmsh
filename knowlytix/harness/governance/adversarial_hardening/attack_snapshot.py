# SPDX-License-Identifier: Apache-2.0
"""Deterministic snapshot of an adversarial-suite run.

The canonical Ch.18 suite (`AdversarialSuite.run_all()`) returns
five `AttackResult`s. The `attack_name`, `blocked`, and
`defense_layer` fields are the audit-grade evidence; `mechanism`
and `evidence` carry per-build details that may include line
numbers or transient identifiers.

This module captures the deterministic view as an
`AttackSnapshot` with a SHA-256 fingerprint over the canonical
JSON. The runtime runs the suite, takes a snapshot, and gates
against the previously-blessed fingerprint. Any defense
regression (blocked->unblocked, layer change) flips the
fingerprint and is caught at ship time.

Mirrors :mod:`governed_agent.workflow_hardening.workflow_snapshot`
for the adversarial surface.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowlytix.harness.governance.agents.adversarial import AttackResult


@dataclass
class AttackOutcome:
    """Deterministic per-attack record (no per-build evidence)."""

    attack_name: str
    blocked: bool
    defense_layer: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_attack_result(cls, r: "AttackResult") -> "AttackOutcome":
        return cls(
            attack_name=r.attack_name,
            blocked=r.blocked,
            defense_layer=r.defense_layer,
        )


@dataclass
class AttackSnapshot:
    """Audit-grade summary of an adversarial-suite execution."""

    suite_id: str
    outcomes: list[AttackOutcome]
    n_attacks: int
    n_blocked: int
    defense_rate: float                   # rounded to 6 decimals

    def canonical_payload(self) -> bytes:
        d = {
            "suite_id": self.suite_id,
            "outcomes": sorted(
                [o.to_dict() for o in self.outcomes],
                key=lambda d_: d_["attack_name"],
            ),
            "n_attacks": self.n_attacks,
            "n_blocked": self.n_blocked,
            "defense_rate": self.defense_rate,
        }
        return json.dumps(d, sort_keys=True).encode("utf-8")

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_payload()).hexdigest()

    def summary(self) -> str:
        return (
            f"AttackSnapshot(id={self.suite_id}, "
            f"blocked={self.n_blocked}/{self.n_attacks}, "
            f"rate={self.defense_rate}, "
            f"fingerprint={self.fingerprint()[:16]}...)"
        )


def build_attack_snapshot(
    *, suite_id: str, results: list,
) -> AttackSnapshot:
    """Pack `AttackResult`s into a deterministic snapshot."""
    outcomes = [AttackOutcome.from_attack_result(r) for r in results]
    n = len(outcomes)
    n_blocked = sum(1 for o in outcomes if o.blocked)
    rate = round(n_blocked / n, 6) if n else 0.0
    return AttackSnapshot(
        suite_id=suite_id,
        outcomes=outcomes,
        n_attacks=n,
        n_blocked=n_blocked,
        defense_rate=rate,
    )
