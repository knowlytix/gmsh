# SPDX-License-Identifier: Apache-2.0
"""Independent second-implementation re-derivation of attack verdicts.

Mirrors `policy_validator_alt`, `audit_alt_validator`, and
`workflow_alt_validator` for the adversarial surface. The
primary `AdversarialSuite` produces an `AttackResult` with
`blocked` and `defense_layer` fields; this alt-validator
re-derives the verdict from a string-table lookup over
`(attack_name, defense_layer)` so disagreement reveals a real
divergence between the two implementations.

Truth table:

    blocked  iff  defense_layer != "none"

If the primary's `blocked` flag does not align with the layer
field, the deployment has a logic divergence somewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AltAttackVerdict:
    attack_name: str
    primary_blocked: bool
    alt_blocked: bool
    primary_layer: str
    agreement: bool


@dataclass
class AltAttackAgreementReport:
    n_attacks: int
    n_agree: int
    disagreements: list[AltAttackVerdict] = field(default_factory=list)

    @property
    def agreement_rate(self) -> float:
        if self.n_attacks == 0:
            return 0.0
        return self.n_agree / self.n_attacks

    @property
    def all_agree(self) -> bool:
        return not self.disagreements

    def summary(self) -> str:
        rows = [
            "AltAttackAgreementReport(",
            f"  n_attacks={self.n_attacks}",
            f"  n_agree={self.n_agree}",
            f"  agreement_rate={self.agreement_rate:.4f}",
            f"  disagreements={len(self.disagreements)}",
        ]
        for d in self.disagreements[:5]:
            rows.append(
                f"    {d.attack_name}: primary={d.primary_blocked} "
                f"alt={d.alt_blocked} (layer={d.primary_layer})"
            )
        rows.append(")")
        return "\n".join(rows)


def alt_blocked(defense_layer: str) -> bool:
    """Re-derive the blocked flag from defense_layer alone.

    The primary implementation sets `blocked` and `defense_layer`
    independently; the alt's invariant is that they must
    co-vary --- a layer of "none" means not blocked, anything
    else means blocked.
    """
    return defense_layer != "none" and defense_layer != ""


def assert_attack_alt_agreement(
    results: list,
) -> AltAttackAgreementReport:
    """Run primary + alt-validator on every attack; report agreement."""
    n_total = 0
    n_agree = 0
    disagreements: list[AltAttackVerdict] = []

    for r in results:
        n_total += 1
        alt = alt_blocked(r.defense_layer)
        agree = alt == r.blocked
        if agree:
            n_agree += 1
        else:
            disagreements.append(AltAttackVerdict(
                attack_name=r.attack_name,
                primary_blocked=r.blocked,
                alt_blocked=alt,
                primary_layer=r.defense_layer,
                agreement=agree,
            ))

    return AltAttackAgreementReport(
        n_attacks=n_total,
        n_agree=n_agree,
        disagreements=disagreements,
    )
