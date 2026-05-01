# SPDX-License-Identifier: Apache-2.0
"""GMSH Cross-Document Conflict Testing — inter-store tension analysis.

Detects factual conflicts between multiple document sources by
computing ``tension_energy`` between GMS stores for shared entities.

- Ingest multiple documents into separate GMS stores
- Find shared entities across stores
- For each shared entity, compute inter-store ``tension_energy`` —
  high tension indicates a factual conflict between documents
- Generate questions that target conflicting claims

Usage::

    analyzer = CrossDocumentAnalyzer(
        stores={"doc_a": store_a, "doc_b": store_b},
    )
    conflicts = analyzer.find_conflicts()
    for c in conflicts:
        print(c.entity, c.inter_store_tension, c.conflict_type)

    gen = CrossDocumentGenerator(stores={"a": store_a, "b": store_b})
    questions = gen.generate(store_a)  # primary store
"""

from __future__ import annotations

import json
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from knowlytix.bench.eval.generators.base import GeneratedQuestion, QuestionGenerator, ANSWER_FORMATS

from .oracles import (
    _load_calibrated_classifiers,
    _load_calibrated_thresholds,
    _load_cross_store_calibrated_classifier,
    _load_cross_store_calibrated_thresholds,
)

if TYPE_CHECKING:
    from knowlytix.doc.store import GMSExpertStore


# Hand-picked fallback used only when no calibration is present.
DEFAULT_TAU_TENSION = 0.5

# Inter-store tension ceiling for band-width normalization.  Parallels
# the ``GEODESIC_CEILING`` in ``hallucination.py``.
TENSION_CEILING = 2.0

# Cross-store calibration class indices (match cross_store_calibration.py).
_CROSS_STORE_CLASS_TO_CONFLICT = {
    0: "minor_variance",   # UNCHANGED
    1: "disagreement",     # MODIFIED
    2: "contradiction",    # CONTRADICTED
}


def _pick_reference_store(
    stores: "dict[str, GMSExpertStore]",
) -> "GMSExpertStore | None":
    for store in stores.values():
        if getattr(store, "store_path", None):
            return store
    return None


def _load_cross_store_from_dir(
    calibration_dir: "str | Path | None",
) -> tuple[dict[str, float], object | None]:
    """Read ``cross_store_calibration.json`` + ``.pkl`` from an explicit
    directory.  Returns ``({}, None)`` if either file is missing or
    malformed.  Used so analyzers can point at a shared calibration
    artifact (e.g. ``lab_stores/ch04_ground_truth``) rather than
    requiring per-store copies.

    Accepts both v1 (``thresholds`` at the top level) and v2
    (``projection_thresholds`` holding the scalar taus from the 1-D
    projection of the 3-D classifier).
    """
    if calibration_dir is None:
        return {}, None
    calibration_dir = Path(calibration_dir)
    json_path = calibration_dir / "cross_store_calibration.json"
    pkl_path = calibration_dir / "cross_store_calibration.pkl"
    thresholds: dict[str, float] = {}
    clf = None
    if json_path.is_file():
        try:
            payload = json.loads(json_path.read_text())
            if isinstance(payload, dict):
                raw = (
                    payload.get("projection_thresholds")
                    or payload.get("thresholds")
                    or {}
                )
                if isinstance(raw, dict):
                    thresholds = {
                        k: float(v) for k, v in raw.items()
                        if k in ("tau_minor_upper", "tau_contra_lower")
                        and isinstance(v, (int, float))
                    }
        except (OSError, json.JSONDecodeError):
            thresholds = {}
    if pkl_path.is_file():
        try:
            with pkl_path.open("rb") as f:
                clf = pickle.load(f)
        except Exception:
            clf = None
    return thresholds, clf


# ---------------------------------------------------------------------------
# Triple-level feature extraction
# ---------------------------------------------------------------------------


def _amended_indices(
    store: "GMSExpertStore",
) -> tuple[set[str], dict[tuple[str, str], set[str]]]:
    """Pre-compute ``(entity_vocab, (h,r)->tails)`` for a store.

    The 3-feature vector for every base triple consults these two
    indices; building them once per store-pair avoids an O(|T|^2) blow
    up inside :meth:`CrossDocumentAnalyzer.find_conflicts`.
    """
    entities: set[str] = set()
    if store.adapter is not None:
        entities = set(store.adapter.entity_to_idx)
    hr_tails: dict[tuple[str, str], set[str]] = {}
    if store.doc_graph is not None:
        for h, r, t in store.doc_graph.triples:
            hr_tails.setdefault((h, r), set()).add(str(t))
    return entities, hr_tails


def _compute_triple_features(
    base_store: "GMSExpertStore",
    amended_store: "GMSExpertStore",
    h: str, r: str, t: str,
    *,
    amended_entities: set[str],
    amended_hr_tails: dict[tuple[str, str], set[str]],
) -> tuple[float, int, int] | None:
    """Compute ``(delta, tail_missing, has_replacement)`` for one triple.

    Returns ``None`` when ``base_store`` cannot score the triple (no
    signal to classify).  Mirrors the feature extraction in
    :func:`harness.doe.cross_store_calibration.label_cross_store_tension`
    so training features and inference features come from the same
    generator.
    """
    t_str = str(t)
    tail_missing = 0 if t_str in amended_entities else 1
    other_tails = amended_hr_tails.get((h, r), set())
    has_replacement = 1 if any(tt != t_str for tt in other_tails) else 0

    try:
        s_base = base_store.score_triple(h, r, t)
    except Exception:
        s_base = None
    if s_base is None:
        return None
    s_amend: float | None = None
    if tail_missing == 0:
        try:
            s_amend_raw = amended_store.score_triple(h, r, t)
        except Exception:
            s_amend_raw = None
        s_amend = None if s_amend_raw is None else float(s_amend_raw)
    delta = (abs(float(s_base) - float(s_amend))
             if s_amend is not None else 0.0)
    return (float(delta), int(tail_missing), int(has_replacement))


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DocumentConflict:
    """A detected conflict between two document stores.

    Attributes
    ----------
    entity : str
        The shared entity where conflict is detected.
    store_a_id : str
        Identifier for the first store.
    store_b_id : str
        Identifier for the second store.
    fact_a : str
        Fact/relation context from store A.
    fact_b : str
        Fact/relation context from store B.
    inter_store_tension : float
        Tension energy between store representations.
        Higher = stronger conflict.
    conflict_type : str
        Classification: ``"contradiction"``, ``"disagreement"``,
        ``"minor_variance"``.  Band boundaries are anchored on the
        calibrated :math:`\\tau_{\\text{tension}}` and the
        :data:`TENSION_CEILING` via :func:`_conflict_type` --
        **not** hard-coded constants, so a recalibration of the judge
        propagates through the entire presentation layer.
    confidence : float
        :math:`P(\\text{verdict correct})` under the calibrated
        ``tension`` classifier, or a hand-mapped fallback when
        ``calibration.pkl`` is missing.  High values mean the
        ``conflict_type`` label is reliable.
    """

    entity: str = ""
    relation: str = ""
    tail: str = ""
    store_a_id: str = ""
    store_b_id: str = ""
    fact_a: str = ""
    fact_b: str = ""
    inter_store_tension: float = 0.0
    tail_missing: int = 0
    has_replacement: int = 0
    conflict_type: str = "minor_variance"
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "entity": self.entity,
            "relation": self.relation,
            "tail": self.tail,
            "stores": [self.store_a_id, self.store_b_id],
            "inter_store_tension": round(self.inter_store_tension, 4),
            "tail_missing": int(self.tail_missing),
            "has_replacement": int(self.has_replacement),
            "conflict_type": self.conflict_type,
            "fact_a": self.fact_a,
            "fact_b": self.fact_b,
            "confidence": round(self.confidence, 4),
        }


def _conflict_type(
    tension: float,
    tau: float | None = None,
    *,
    tau_minor_upper: float | None = None,
    tau_contra_lower: float | None = None,
) -> str:
    """Classify conflict severity by inter-store tension.

    Preference order (highest priority first):

    1. **Cross-store calibrated thresholds**
       (``tau_minor_upper`` and ``tau_contra_lower`` from
       :mod:`harness.doe.cross_store_calibration`).  These are learned
       from a labeled perturbation pair, so the bands are data-driven:

       - ``minor_variance``  -- ``t <= tau_minor_upper``
       - ``disagreement``    -- ``tau_minor_upper < t <= tau_contra_lower``
       - ``contradiction``   -- ``t > tau_contra_lower``

    2. **Per-claim tension anchor** (``tau``), with
       :math:`\\Delta = (\\text{ceiling} - \\tau) / 3`.  Used only when
       no cross-store calibration is available; per-claim and inter-store
       signals live on different scales, so this is a degraded fallback.

       - ``minor_variance``  -- ``t <= tau``
       - ``disagreement``    -- ``tau < t <= tau + 2*Delta``
       - ``contradiction``   -- ``t > tau + 2*Delta``

    3. **Historical constants** (``0.7``, ``1.5``) when neither a
       ``tau`` nor cross-store thresholds are available.
    """
    # Prefer cross-store calibrated bands when available.
    if (
        tau_minor_upper is not None and tau_contra_lower is not None
        and tau_minor_upper >= 0 and tau_contra_lower >= tau_minor_upper
    ):
        if tension <= tau_minor_upper:
            return "minor_variance"
        if tension <= tau_contra_lower:
            return "disagreement"
        return "contradiction"
    if tau is None or tau <= 0:
        if tension > 1.5:
            return "contradiction"
        if tension > 0.7:
            return "disagreement"
        return "minor_variance"
    gap = max(TENSION_CEILING - tau, 1e-6)
    delta = gap / 3.0
    if tension <= tau:
        return "minor_variance"
    if tension <= tau + 2.0 * delta:
        return "disagreement"
    return "contradiction"


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class CrossDocumentAnalyzer:
    """Detect factual conflicts between GMS stores using tension_energy.

    Parameters
    ----------
    stores : dict[str, GMSExpertStore]
        Named GMS stores, each from a different document.
    tension_threshold : float, optional
        Minimum tension to flag as a conflict.  ``None`` (default)
        reads :math:`\\tau_{\\text{tension}}` from any store's
        ``calibration.json``; if no calibration is present, falls back
        to :data:`DEFAULT_TAU_TENSION`.
    """

    def __init__(
        self,
        stores: dict[str, "GMSExpertStore"] | None = None,
        tension_threshold: float | None = None,
        *,
        calibration_dir: "str | Path | None" = None,
    ):
        self.stores = stores or {}
        ref_store = _pick_reference_store(self.stores)
        # --- per-claim calibration (legacy fallback)
        calibrated = _load_calibrated_thresholds(ref_store)
        classifiers = _load_calibrated_classifiers(ref_store)
        self.tension_threshold = (
            tension_threshold
            if tension_threshold is not None
            else calibrated.get("tension", DEFAULT_TAU_TENSION)
        )
        self._calibration_source = (
            "override" if tension_threshold is not None
            else ("calibration.json" if "tension" in calibrated
                  else "default")
        )
        self._tension_classifier = classifiers.get("tension")

        # --- cross-store calibration (preferred when available)
        xs_thresholds, xs_classifier = _load_cross_store_from_dir(
            calibration_dir,
        )
        if not xs_thresholds:
            xs_thresholds = _load_cross_store_calibrated_thresholds(ref_store)
        if xs_classifier is None:
            xs_classifier = _load_cross_store_calibrated_classifier(ref_store)
        self._cross_store_thresholds = xs_thresholds or {}
        self._cross_store_classifier = xs_classifier
        if xs_thresholds and calibration_dir is not None:
            self._cross_calibration_source = f"dir:{os.fspath(calibration_dir)}"
        elif xs_thresholds:
            self._cross_calibration_source = "store:cross_store_calibration.json"
        else:
            self._cross_calibration_source = "none"

        # Effective filter threshold: when cross-store thresholds exist,
        # emit any triple whose delta exceeds tau_minor_upper (which is
        # the data-driven UNCHANGED/MODIFIED crossing).  Otherwise keep
        # the legacy per-claim tau.
        self._emit_threshold: float = (
            float(self._cross_store_thresholds.get("tau_minor_upper"))
            if "tau_minor_upper" in self._cross_store_thresholds
            else float(self.tension_threshold)
        )

    def find_shared_entities(self) -> dict[str, list[str]]:
        """Find entities that appear in multiple stores.

        Returns
        -------
        dict[str, list[str]]
            Entity → list of store IDs containing that entity.
        """
        entity_stores: dict[str, list[str]] = {}
        for store_id, store in self.stores.items():
            if store.adapter is None:
                continue
            for entity in store.adapter.entity_to_idx:
                entity_stores.setdefault(entity, []).append(store_id)

        return {
            e: sids for e, sids in entity_stores.items()
            if len(sids) >= 2
        }

    def find_conflicts(
        self, max_conflicts: int = 50,
    ) -> list[DocumentConflict]:
        """Scan each store's triples against every other store and emit
        one :class:`DocumentConflict` per triple whose inter-store delta
        clears the emit threshold.

        Per-triple is the right granularity: a single entity may have
        many triples of which only some are perturbed, and the
        calibrated classifier expects a 3-feature vector
        ``(delta, tail_missing, has_replacement)`` per triple, not a
        scalar per entity.  Classification uses the full 3-D classifier;
        scalar :data:`tau_minor_upper` / :data:`tau_contra_lower` remain
        available as an informational fallback.

        Parameters
        ----------
        max_conflicts : int
            Maximum number of conflicts to return.

        Returns
        -------
        list[DocumentConflict]
            Detected conflicts sorted by tension (descending).
        """
        conflicts: list[DocumentConflict] = []
        tau_minor = self._cross_store_thresholds.get("tau_minor_upper")
        tau_contra = self._cross_store_thresholds.get("tau_contra_lower")

        store_ids = list(self.stores.keys())
        for i, sid_a in enumerate(store_ids):
            store_a = self.stores[sid_a]
            if store_a.doc_graph is None:
                continue
            for sid_b in store_ids[i + 1:]:
                store_b = self.stores[sid_b]
                if store_b.doc_graph is None:
                    continue
                # Precompute amended-side indices once per pair.
                b_entities, b_hr_tails = _amended_indices(store_b)
                for h, r, t in store_a.doc_graph.triples:
                    features = _compute_triple_features(
                        store_a, store_b, h, r, t,
                        amended_entities=b_entities,
                        amended_hr_tails=b_hr_tails,
                    )
                    if features is None:
                        continue
                    delta, tail_missing, has_replacement = features
                    ctype = self._classify_features(
                        delta, tail_missing, has_replacement,
                        tau_minor=tau_minor, tau_contra=tau_contra,
                    )
                    # Emit gate: drop clearly-UNCHANGED triples.  When the
                    # 3-D classifier is loaded, its prediction is the gate
                    # (so MODIFIED triples whose delta is 0 because the
                    # tail is orphaned are still emitted).  Otherwise fall
                    # back to the scalar delta cut.
                    if self._cross_store_classifier is not None:
                        if ctype == "minor_variance":
                            continue
                    else:
                        if delta < self._emit_threshold:
                            continue
                    conf = self._conflict_confidence(
                        (delta, tail_missing, has_replacement), ctype,
                    )
                    fact_a = f"{h} {r} {t}"
                    if tail_missing:
                        fact_b = f"(tail {t!r} absent from {sid_b})"
                    else:
                        fact_b = f"({h} {r} {t}) present in {sid_b}"
                    conflicts.append(DocumentConflict(
                        entity=h,
                        relation=r,
                        tail=str(t),
                        store_a_id=sid_a,
                        store_b_id=sid_b,
                        fact_a=fact_a,
                        fact_b=fact_b,
                        inter_store_tension=float(delta),
                        tail_missing=int(tail_missing),
                        has_replacement=int(has_replacement),
                        conflict_type=ctype,
                        confidence=float(conf),
                    ))

        conflicts.sort(key=lambda c: c.inter_store_tension, reverse=True)
        return conflicts[:max_conflicts]

    def _classify_features(
        self,
        delta: float,
        tail_missing: int,
        has_replacement: int,
        *,
        tau_minor: float | None,
        tau_contra: float | None,
    ) -> str:
        """Pick the conflict label for one 3-D feature vector.

        Prefers the calibrated 3-D classifier (bank-grade path).  Falls
        back to the scalar :func:`_conflict_type` band rules when no
        classifier is loaded, so a library-only install still reports
        useful labels.
        """
        if self._cross_store_classifier is not None:
            try:
                X = [[float(delta), float(tail_missing), float(has_replacement)]]
                cls_idx = int(self._cross_store_classifier.predict(X)[0])
                return _CROSS_STORE_CLASS_TO_CONFLICT.get(
                    cls_idx, "minor_variance",
                )
            except Exception:
                pass
        return _conflict_type(
            float(delta), self.tension_threshold,
            tau_minor_upper=tau_minor, tau_contra_lower=tau_contra,
        )

    def _conflict_confidence(
        self,
        tension: "float | tuple[float, int, int] | list[float]",
        conflict_type: str,
    ) -> float:
        """Return :math:`P(\\text{verdict correct})` for the conflict
        classification.

        Parameters
        ----------
        tension
            Either a scalar delta (backward compat with pedagogical
            notebook cells that pass a single float) or the full 3-D
            feature vector ``(delta, tail_missing, has_replacement)``.
            When a scalar is supplied, ``tail_missing`` and
            ``has_replacement`` default to ``(0, 1)`` -- the cell where
            most non-UNCHANGED examples live, so the scalar path remains
            informative.
        conflict_type
            The predicted label whose probability is returned.

        Preference order:

        1. **Cross-store calibrated classifier** -- returns
           :math:`P(\\text{class} = \\text{predicted label})` directly
           from the isotonic-calibrated 3-class classifier.  When the
           caller passes a scalar, the missing features are filled with
           the projection cell used by :func:`_project_delta_thresholds`.
        2. **Per-claim tension classifier** (``calibration.pkl``).
           Scalar-only; if a vector is supplied its first element is
           used.  ``minor_variance`` maps to ``passed``.
        3. **Hand-mapped fallback**
           :math:`\\max(0, 1 - t / (2 \\tau_{\\text{tension}}))`.
        """
        if isinstance(tension, (tuple, list)):
            delta, tail_missing, has_replacement = (
                float(tension[0]),
                int(tension[1]) if len(tension) > 1 else 0,
                int(tension[2]) if len(tension) > 2 else 1,
            )
        else:
            delta = float(tension)
            tail_missing, has_replacement = 0, 1

        # 1. Cross-store calibrated classifier (preferred)
        if self._cross_store_classifier is not None:
            try:
                X = [[delta, float(tail_missing), float(has_replacement)]]
                probs = self._cross_store_classifier.predict_proba(X)[0]
                inv = {v: k for k, v in _CROSS_STORE_CLASS_TO_CONFLICT.items()}
                cls_idx = inv.get(conflict_type)
                if cls_idx is not None and 0 <= cls_idx < len(probs):
                    return float(probs[cls_idx])
            except Exception:
                pass
        # 2. Per-claim tension classifier (legacy, scalar only)
        tau = max(float(self.tension_threshold), 1e-6)
        passed = conflict_type == "minor_variance"
        if self._tension_classifier is not None:
            try:
                p_real = float(
                    self._tension_classifier.predict_proba([[delta]])[0, 1]
                )
                return p_real if passed else 1.0 - p_real
            except Exception:
                pass
        # 3. Hand-mapped fallback
        return max(0.0, min(1.0, 1.0 - delta / (2.0 * tau)))

    def _compute_tension(
        self,
        store_a: "GMSExpertStore",
        store_b: "GMSExpertStore",
        entity: str,
    ) -> float | None:
        """Entity-level scalar tension, preserved for backward compat.

        Aggregates the per-triple deltas of triples headed by
        ``entity`` in ``store_a`` and returns the max.  Replaces the
        legacy ``score_triple(entity, "is_type", entity)`` path, which
        silently returned ``None`` whenever ``"is_type"`` was not in the
        store's relation vocabulary (the common case).
        """
        if store_a.doc_graph is None or store_b.doc_graph is None:
            return None
        b_entities, b_hr_tails = _amended_indices(store_b)
        deltas: list[float] = []
        for h, r, t in store_a.doc_graph.triples:
            if h != entity and t != entity:
                continue
            features = _compute_triple_features(
                store_a, store_b, h, r, t,
                amended_entities=b_entities,
                amended_hr_tails=b_hr_tails,
            )
            if features is None:
                continue
            deltas.append(features[0])
        if not deltas:
            return None
        return float(max(deltas))

    def _entity_context(
        self, store: "GMSExpertStore", entity: str,
    ) -> str:
        """Get a brief fact summary for an entity in a store."""
        if store.doc_graph is None:
            return ""
        triples = [
            (h, r, t) for h, r, t in store.doc_graph.triples
            if h == entity or t == entity
        ]
        if not triples:
            return f"'{entity}' present but no triples found"
        summaries = [f"{h} {r} {t}" for h, r, t in triples[:3]]
        return "; ".join(summaries)


# ---------------------------------------------------------------------------
# Question generator
# ---------------------------------------------------------------------------

class CrossDocumentGenerator(QuestionGenerator):
    """Generate questions targeting cross-document conflicts.

    Requires multiple stores to detect conflicts.  Falls back to
    intra-store tension for single-store scenarios.

    Category: ``"cross_document"``

    Parameters
    ----------
    stores : dict[str, GMSExpertStore] or None
        Additional named stores for cross-document analysis.
    """

    category = "cross_document"

    def __init__(
        self, stores: dict[str, "GMSExpertStore"] | None = None,
    ):
        self.additional_stores = stores or {}

    def generate(self, store: "GMSExpertStore") -> list[GeneratedQuestion]:
        if store.doc_graph is None or store.adapter is None:
            return []

        questions: list[GeneratedQuestion] = []
        rng = random.Random(42)

        # If we have additional stores, find conflicts
        if self.additional_stores:
            all_stores = {"primary": store, **self.additional_stores}
            analyzer = CrossDocumentAnalyzer(stores=all_stores)
            conflicts = analyzer.find_conflicts(max_conflicts=20)

            for i, conflict in enumerate(conflicts[:10]):
                ground_truth = conflict.conflict_type

                def _answer_fn(s, _ent=conflict.entity, _type=conflict.conflict_type):
                    return _type

                questions.append(GeneratedQuestion(
                    qid=f"cross_doc_{i:03d}",
                    category=self.category,
                    natural_language=(
                        f"Document A says: {conflict.fact_a}. "
                        f"Document B says: {conflict.fact_b}. "
                        f"Are these consistent?"
                    ),
                    gms_answer_fn=_answer_fn,
                    ground_truth=ground_truth,
                    llm_prompt=(
                        f"Compare these two claims about '{conflict.entity}':\n"
                        f"Source A: {conflict.fact_a}\n"
                        f"Source B: {conflict.fact_b}\n\n"
                        f"Do these sources agree? Classify as: "
                        f"'minor_variance', 'disagreement', or "
                        f"'contradiction'.\n\n"
                        f"ANSWER: <classification>"
                    ),
                    answer_type="str",
                    metadata={
                        "entity": conflict.entity,
                        "tension": round(conflict.inter_store_tension, 4),
                        "test_type": "cross_document_conflict",
                    },
                ))
        else:
            # Single-store: find entities with internal tension
            triples = list(store.doc_graph.triples)
            entities = list(store.adapter.entity_to_idx.keys())

            for i, ent in enumerate(entities[:15]):
                ent_triples = [
                    (h, r, t) for h, r, t in triples
                    if h == ent
                ]
                if len(ent_triples) < 2:
                    continue

                # Check tension between different tails
                t1, t2 = ent_triples[0], ent_triples[1]
                tension = None
                try:
                    tension = store.tension_energy(t1[2], t2[2])
                except Exception:
                    pass

                if tension is None:
                    continue

                ground_truth = tension < 0.7  # consistent if low tension

                def _answer_fn(s, _gt=ground_truth):
                    return _gt

                questions.append(GeneratedQuestion(
                    qid=f"cross_doc_intra_{i:03d}",
                    category=self.category,
                    natural_language=(
                        f"Are the facts '{t1[0]} {t1[1]} {t1[2]}' and "
                        f"'{t2[0]} {t2[1]} {t2[2]}' mutually consistent?"
                    ),
                    gms_answer_fn=_answer_fn,
                    ground_truth=ground_truth,
                    llm_prompt=(
                        f"Based on the document, are these two facts "
                        f"consistent?\n"
                        f"1. {t1[0]} {t1[1]} {t1[2]}\n"
                        f"2. {t2[0]} {t2[1]} {t2[2]}\n\n"
                        f"{ANSWER_FORMATS['bool']}"
                    ),
                    answer_type="bool",
                    metadata={
                        "entity": ent,
                        "tension": round(tension, 4) if tension else None,
                        "test_type": "intra_document_consistency",
                    },
                ))

                if len(questions) >= 10:
                    break

        return questions
