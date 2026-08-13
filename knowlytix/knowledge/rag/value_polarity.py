# SPDX-License-Identifier: Apache-2.0
"""Fused categorical-value verification: v-space identity + cap plausibility + u-space polarity.

A synthesized answer's asserted value for a single-valued relation is checked three
ways, each signal failing in a different place so the fusion beats any one:

  * **cap plausibility** (v-space): ``store.score_triple_emb`` of the value's tuned
    v-embedding against the relation cap -- the exact stored value (or a graph alias)
    lands inside the cap.
  * **entity resolution** (v-space identity): the value's nearest trained entity. A
    synonym resolves onto the stored value; a different value resolves elsewhere.
    Identity, not polarity -- an *antonym* also resolves onto the value's axis.
  * **u-space polarity** (3-class tension): below ``tau_ent`` the value agrees with
    the stored one (synonym), above ``tau_contra`` it contradicts (opposite stance),
    in between it is unrelated/uncertain. The two cuts are calibrated with the
    knowlytix 3-class fitter (5-fold CV gated); the middle band IS the abstention
    region, so a value the geometry cannot resolve is deferred, never guessed.

Numeric values are out of scope (sentence encoders do not separate "120" from "999")
-- the caller keeps the exact/numeric path for those. Categorical/polarity values
(forbidden/permitted, required/optional, issued/denied, yes/no, ...) go here.

The runtime decision is pure geometry; an LLM is used only at BUILD time to
materialize the synonym/antonym training data for the u-encoder SFT and the
calibration cohort (see scripts/build_clean_polarity_sft.py).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

__all__ = ["PolarityCuts", "ValuePolarityChecker", "polarity_tension_cohort"]


def polarity_tension_cohort(u_encoder, axes):
    """Build the 3-class u-space tension cohort for :meth:`PolarityCuts.calibrate`.

    Each axis is ``(pole_a, syns_a, pole_b, syns_b)`` where ``pole_a`` and ``pole_b``
    are opposite stances (e.g. ``forbidden`` / ``permitted``) and the synonym lists
    are same-stance paraphrases. For each pole the function measures u-space tension
    between the pole and every other value and labels the pair by relationship:
    ``0`` agree (a synonym, low tension), ``2`` contradict (the opposite pole or its
    synonyms, high tension), ``1`` unrelated (a value on a different axis, mid). The
    returned ``(tension_values, labels)`` feed the 3-class CV fitter directly.

    The tension computation (encode the pair with the polarity-tuned u-encoder,
    cosine, map to tension) lives here so a caller supplies only the domain axes.
    """
    import torch
    import torch.nn.functional as F

    from knowlytix.knowledge.rag.relevance import _tension

    def uten(a: str, b: str) -> float:
        x = F.normalize(torch.as_tensor(u_encoder([a, b]), dtype=torch.float32),
                        p=2, dim=-1)
        return _tension(float(x[0] @ x[1]))

    all_vals = {p for ax in axes for p in (ax[0], ax[2])}
    tension_values: list[float] = []
    labels: list[int] = []
    for pole_a, syns_a, pole_b, syns_b in axes:
        for anchor, syns, opp, opp_syns in [(pole_a, syns_a, pole_b, syns_b),
                                            (pole_b, syns_b, pole_a, syns_a)]:
            for s in [anchor] + list(syns):
                tension_values.append(uten(anchor, s))
                labels.append(0)   # agree
            for s in [opp] + list(opp_syns):
                tension_values.append(uten(anchor, s))
                labels.append(2)   # contradict
            for other in all_vals - {anchor, opp}:
                tension_values.append(uten(anchor, other))
                labels.append(1)  # unrelated
    return tension_values, labels


@dataclass
class PolarityCuts:
    """Calibrated 3-class u-space tension cuts. Below ``tau_ent`` -> agree
    (supported); above ``tau_contra`` -> contradict; between -> uncertain."""

    tau_ent: float
    tau_contra: float
    cv_accuracy: float | None = None

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({"tau_ent": self.tau_ent, "tau_contra": self.tau_contra,
                       "cv_accuracy": self.cv_accuracy}, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "PolarityCuts | None":
        if not os.path.exists(path):
            return None
        d = json.load(open(path))
        return cls(float(d["tau_ent"]), float(d["tau_contra"]),
                   d.get("cv_accuracy"))

    @staticmethod
    def calibrate(tension_values, labels, *, seed: int = 42) -> "PolarityCuts | None":
        """CV-calibrate the two cuts from 3-class tension samples (0=agree,
        1=unrelated, 2=contradict) via knowlytix ``fit_three_class_thresholds``
        (5-fold stratified CV + accuracy gate). Returns ``None`` when the channel
        is degenerate (CV accuracy below the gate or no two ordered cuts) -- the
        honest signal that u-space tension is not usable on this store, so the
        caller should fall back to cap+resolution only."""
        from knowlytix.harness.testing.threshold_calibration import (
            fit_three_class_thresholds,
        )
        clf, cuts = fit_three_class_thresholds(list(tension_values), list(labels),
                                               seed=seed)
        if cuts is None:
            return None
        return PolarityCuts(float(cuts[0]), float(cuts[1]),
                            getattr(clf, "_cv_accuracy_", None))


class ValuePolarityChecker:
    """Fused value check returning ``supported`` | ``contradicted`` | ``uncertain``.

    Args:
        store: a cap-trained GMSExpertStore (needs ``score_triple_emb``,
            ``cap_radius``, ``fuzzy_match_entity``).
        v_encoder: the document-tuned v-encoder (``out_dim == d_v``) -- the same
            one that warm-started the store, so a value's embedding is in-frame.
        u_encoder: the polarity-tuned u-encoder (``value_polarity_encoder``).
        cuts: calibrated :class:`PolarityCuts`; when ``None`` the u-space gate is
            skipped (degenerate) and only cap + resolution decide.
        res_margin: min nearest-vs-runner-up cosine gap for a decisive resolution.
    """

    def __init__(self, store, v_encoder, u_encoder, cuts: "PolarityCuts | None",
                 *, res_margin: float = 0.08):
        self.store = store
        self._v = v_encoder
        self._u = u_encoder
        self.cuts = cuts
        self.res_margin = res_margin
        self._ents = None
        self._E = None

    # -- lazy entity embedding table for resolution --------------------------
    def _entity_matrix(self):
        if self._E is None:
            import torch
            import torch.nn.functional as F
            self._ents = sorted(self.store.adapter.entity_to_idx)
            self._E = F.normalize(
                torch.as_tensor(self._v(self._ents), dtype=torch.float32), p=2, dim=-1)
        return self._ents, self._E

    def _resolve(self, token: str):
        import torch
        import torch.nn.functional as F
        ents, E = self._entity_matrix()
        q = F.normalize(torch.as_tensor(self._v([token]), dtype=torch.float32),
                        p=2, dim=-1)[0]
        sims = E @ q
        top = torch.topk(sims, min(2, len(ents)))
        margin = float(top.values[0] - top.values[1]) if len(ents) > 1 else 1.0
        return ents[int(top.indices[0])], margin

    def _tension(self, a: str, b: str) -> float:
        import torch
        import torch.nn.functional as F

        from knowlytix.knowledge.rag.relevance import _tension
        v = F.normalize(torch.as_tensor(self._u([a, b]), dtype=torch.float32),
                        p=2, dim=-1)
        return _tension(float(v[0] @ v[1]))

    def check(self, head: str, relation: str, asserted: str, stored: str) -> str:
        """Fused verdict for an asserted categorical value vs the stored one."""
        cap = self.store.cap_radius(relation)
        d = self.store.score_triple_emb(head, relation, self._v([asserted])[0])
        if cap is not None and d is not None and d <= cap:
            return "supported"                       # exact stored value (or alias)
        canon = getattr(self.store, "_canon_entity", lambda x: x)
        stored_e = canon(stored)
        near, margin = self._resolve(asserted)
        near_stored = near == stored_e
        if margin >= self.res_margin and not near_stored:
            return "contradicted"                    # resolves to a different value
        if self.cuts is None:
            # u-space degenerate: trust resolution only.
            return "supported" if near_stored else "uncertain"
        u = self._tension(asserted, stored)
        if u <= self.cuts.tau_ent:
            return "supported" if near_stored else "uncertain"
        if u >= self.cuts.tau_contra:
            return "contradicted"                    # opposite polarity
        return "uncertain"                           # unrelated middle band -> defer
