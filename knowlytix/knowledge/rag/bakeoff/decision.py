# SPDX-License-Identifier: Apache-2.0
"""Bake-off decision rule and audit record (spec Section 6, gate G4).

The crossover report says which binder leads where; the decision rule turns that
into a governed recommendation against stated ceilings and an audit record. A
binder is *admissible* only if it clears every ceiling:

* ``mis_bind_rate`` -- confident-wrong bindings, at or below the ceiling (the
  load-bearing safety bound);
* ``conformal_coverage`` -- the holdout prediction-set coverage, at or above the
  ceiling (calibrated abstention actually covers the truth);
* ``synthetic_vs_real_gap`` -- the accuracy drop from the synthetic cohort to the
  real-language holdout, at or below the ceiling (it did not just memorize the DOE).

Among admissible binders the rule prefers the one that generalizes -- highest
real-language (holdout) accuracy -- breaking ties toward the cheaper patch
mechanism, then higher overall accuracy. If no binder clears every ceiling the rule
RECUSES from a single pick (``recommended`` is ``None``) and reports the nearest
binder with its violations, rather than shipping a binder that fails a bound.

The ceilings here are un-ratified placeholders until model risk management sets
them (spec Section 10); ``ceilings_ratified`` records that in the audit trail.
"""

from __future__ import annotations

__all__ = ["DEFAULT_CEILINGS", "decide"]

# Placeholder operating ceilings (spec Section 10), un-ratified until MRM sets them.
DEFAULT_CEILINGS = {
    "mis_bind_rate": 0.05,          # <= (lower is safer)
    "conformal_coverage": 0.90,     # >= (higher covers the truth)
    "synthetic_vs_real_gap": 0.10,  # <= (smaller means it generalized)
}


def _arm_metrics(entry: dict) -> dict:
    """Pull the decision-relevant scalars out of one arm's report entry."""
    overall = entry.get("overall", {})
    holdout = entry.get("holdout", {})
    conformal = entry.get("conformal", {})
    return {
        "accuracy": overall.get("accuracy"),
        "mis_bind_rate": overall.get("mis_bind_rate"),
        "oos_false_accept": overall.get("oos_false_accept"),
        "novel_entity_score": entry.get("novel_entity", {}).get("novel_entity_score"),
        "holdout_accuracy": holdout.get("accuracy"),
        "synthetic_vs_real_gap": entry.get("synthetic_vs_real_gap"),
        "conformal_coverage": conformal.get("coverage"),
        "patch_cost": entry.get("patch_cost"),
        "patch_mechanism": entry.get("patch_mechanism"),
    }


def _check_ceilings(m: dict, ceilings: dict) -> dict:
    """Per-ceiling pass/fail for one arm; a missing metric fails (cannot certify)."""
    checks = {}
    if "mis_bind_rate" in ceilings:
        v = m.get("mis_bind_rate")
        checks["mis_bind_rate"] = v is not None and v <= ceilings["mis_bind_rate"]
    if "conformal_coverage" in ceilings:
        v = m.get("conformal_coverage")
        checks["conformal_coverage"] = v is not None and v >= ceilings["conformal_coverage"]
    if "synthetic_vs_real_gap" in ceilings:
        v = m.get("synthetic_vs_real_gap")
        checks["synthetic_vs_real_gap"] = v is not None and v <= ceilings["synthetic_vs_real_gap"]
    return checks


def _rank_key(m: dict):
    """Prefer higher real-language accuracy, then cheaper patch, then overall accuracy.
    None sorts last."""
    def _num(x, default):
        return x if isinstance(x, (int, float)) else default
    return (_num(m.get("holdout_accuracy"), -1.0),
            -_num(m.get("patch_cost"), 99),
            _num(m.get("accuracy"), -1.0))


def decide(report: dict, *, ceilings: dict | None = None, ratified: bool = False,
           timestamp: str | None = None) -> dict:
    """Apply the G4 decision rule to a :func:`bakeoff` report.

    Args:
        report: the ``{"arms", "crossover"}`` dict from :func:`bakeoff`.
        ceilings: admissibility ceilings; defaults to :data:`DEFAULT_CEILINGS`.
        ratified: whether ``ceilings`` are MRM-ratified (recorded in the audit).
        timestamp: an ISO timestamp for the audit record (the workflow stamps it;
            the runtime has no clock here).

    Returns a dict with ``ceilings``, ``ceilings_ratified``, per-arm ``arms``
    (metrics + ceiling ``checks`` + ``admissible``), the ``recommended`` arm (or
    ``None`` when the rule recuses), a ``rationale``, the per-factor
    ``by_regime`` leaders, and an ``audit`` record (I4).
    """
    ceilings = dict(ceilings or DEFAULT_CEILINGS)
    arms_in = report.get("arms", {})

    arms: dict[str, dict] = {}
    for name, entry in arms_in.items():
        m = _arm_metrics(entry)
        checks = _check_ceilings(m, ceilings)
        arms[name] = {"metrics": m, "checks": checks, "admissible": all(checks.values())}

    admissible = [n for n, a in arms.items() if a["admissible"]]
    if admissible:
        recommended = max(admissible, key=lambda n: _rank_key(arms[n]["metrics"]))
        m = arms[recommended]["metrics"]
        rationale = (
            f"{recommended} clears every ceiling and leads the admissible set on "
            f"real-language accuracy ({m.get('holdout_accuracy')}) at patch cost "
            f"{m.get('patch_cost')} ({m.get('patch_mechanism')}).")
        recuse = False
    else:
        recommended = None
        # Nearest = fewest failed ceilings, then best real-language accuracy.
        nearest = min(arms, key=lambda n: (
            sum(1 for ok in arms[n]["checks"].values() if not ok),
            -_rank_key(arms[n]["metrics"])[0])) if arms else None
        fails = [c for c, ok in (arms.get(nearest, {}).get("checks", {}) or {}).items()
                 if not ok]
        rationale = (
            "No binder clears every ceiling; recusing from a single pick. Nearest is "
            f"{nearest}, failing {fails or '(no arms)'}. The per-regime crossover still "
            "identifies a leader within each factor level.")
        recuse = True

    by_regime = {
        f: {lv: cell.get("accuracy_leader") for lv, cell in levels.items()}
        for f, levels in report.get("crossover", {}).get("by_factor", {}).items()
    }

    audit = {
        "timestamp": timestamp,
        "ceilings": ceilings,
        "ceilings_ratified": ratified,
        "overall_accuracy_leader": report.get("crossover", {}).get("overall_accuracy_leader"),
        "per_arm": {n: {**a["metrics"], "checks": a["checks"],
                        "admissible": a["admissible"]} for n, a in arms.items()},
        "recommended": recommended,
        "recused": recuse,
        "rationale": rationale,
    }
    return {
        "ceilings": ceilings,
        "ceilings_ratified": ratified,
        "arms": arms,
        "recommended": recommended,
        "recused": recuse,
        "rationale": rationale,
        "by_regime": by_regime,
        "audit": audit,
    }
