# SPDX-License-Identifier: Apache-2.0
"""GMSH Truth Objects — typed truth extracted from documents and linked to GMS.

Every truth object connects to a specific GMS primitive:

- EntityTruth   → entity in adapter.entity_to_idx (embedding vector)
- RelationTruth → triple in doc_graph.triples (scored via score_triple)
- ThresholdTruth → phase_encoder in doc_graph.phase_encoders
- ArtifactTruth  → ENM entry with SHA-256 hash

TruthStore scans a populated GMSExpertStore and creates typed truths,
providing a structured registry that the ClaimCompiler can reference
for provenance attachment.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore

    from .claims import ClaimBase


# ---------------------------------------------------------------------------
# Truth data models
# ---------------------------------------------------------------------------

@dataclass
class TruthBase:
    """Base class for all typed truth objects."""

    truth_id: str = ""
    doc_id: str = ""           # source document hash
    section_tag: str = ""      # normalized section name
    confidence: float = 1.0
    truth_type: str = "base"

    def __post_init__(self):
        if not self.truth_id:
            self.truth_id = uuid.uuid4().hex[:12]


@dataclass
class EntityTruth(TruthBase):
    """A known entity grounded in the GMS embedding space.

    Links to: ``adapter.entity_to_idx[entity]``
    """

    entity: str = ""
    entity_type: str = ""      # optional type label
    gms_idx: int = -1          # index in adapter.entity_to_idx
    truth_type: str = "entity"


@dataclass
class RelationTruth(TruthBase):
    """A known (head, relation, tail) triple scored against the manifold.

    Links to: ``store.score_triple(head, relation, tail)``
    """

    head: str = ""
    relation: str = ""
    tail: str = ""
    gms_score: float | None = None  # geodesic distance at ingest time
    truth_type: str = "relation"


@dataclass
class ThresholdTruth(TruthBase):
    """A known numeric threshold from a phase encoder.

    Links to: ``doc_graph.phase_encoders[encoder_name]``
    """

    entity: str = ""
    metric: str = ""           # encoder name
    value: float = 0.0
    encoder_name: str = ""
    phase_angle: float = 0.0   # φ(value)
    truth_type: str = "threshold"


@dataclass
class ArtifactTruth(TruthBase):
    """A known exact numeric value from ENM.

    Links to: ``store.lookup_enm(category, entity_id)``
    """

    category: str = ""
    entity_id: str = ""
    exact_value: float = 0.0
    enm_hash: str = ""         # SHA-256 of the stored value
    truth_type: str = "artifact"


# ---------------------------------------------------------------------------
# TruthStore
# ---------------------------------------------------------------------------

class TruthStore:
    """Registry of typed truths extracted from a GMS store.

    Built during ingestion by scanning the populated GMSExpertStore.
    Each truth links to a specific GMS primitive (embedding, rotor,
    ENM entry, phase encoder).

    Parameters
    ----------
    store : GMSExpertStore
        The populated GMS store to scan.
    """

    def __init__(self, store: "GMSExpertStore", tracing=None):
        self.store = store
        self.tracing = tracing  # DOETracingGMS for recording build events
        self._truths: dict[str, TruthBase] = {}

        # Indices for fast lookup
        self._by_entity: dict[str, list[str]] = {}     # entity -> [truth_id]
        self._by_relation: dict[str, list[str]] = {}    # "h|r|t" -> [truth_id]
        self._by_enm_key: dict[str, str] = {}           # "cat|id" -> truth_id
        self._by_encoder: dict[str, list[str]] = {}     # encoder_name -> [truth_id]

    def build_from_store(self):
        """Scan the GMS store and create typed truth objects.

        Creates:
        - EntityTruth for each entity in adapter.entity_to_idx
        - RelationTruth for each triple in doc_graph.triples (with geodesic score)
        - ArtifactTruth for each ENM entry (with SHA-256 hash)
        - ThresholdTruth for each value that has a phase encoder
        """
        import time as _time
        t0 = _time.time()
        store = self.store

        # 1. Entity truths
        if store.adapter:
            for entity, idx in store.adapter.entity_to_idx.items():
                t = EntityTruth(
                    entity=entity,
                    gms_idx=idx,
                )
                self._register(t)
                self._by_entity.setdefault(entity, []).append(t.truth_id)

        # 2. Relation truths (with geodesic scoring)
        if store.doc_graph:
            for h, r, tail in store.doc_graph.triples:
                if r == "in_section":
                    continue  # skip structural triples
                gms_score = store.score_triple(h, r, tail)
                t = RelationTruth(
                    head=h,
                    relation=r,
                    tail=tail,
                    gms_score=gms_score,
                )
                self._register(t)
                key = f"{h}|{r}|{tail}"
                self._by_relation.setdefault(key, []).append(t.truth_id)

        # 3. Artifact truths (ENM entries)
        if store.enm:
            for key in store.enm.keys():
                val = store.enm.read_exact(key)
                if val is None:
                    continue
                fval = float(val.item() if hasattr(val, "item") else val)
                val_hash = hashlib.sha256(
                    str(fval).encode("utf-8")
                ).hexdigest()[:16]
                t = ArtifactTruth(
                    category=key.type,
                    entity_id=key.id,
                    exact_value=fval,
                    enm_hash=val_hash,
                )
                self._register(t)
                enm_key = f"{key.type}|{key.id}"
                self._by_enm_key[enm_key] = t.truth_id

        # 4. Threshold truths (phase encoders + ENM values)
        if store.doc_graph and hasattr(store.doc_graph, "phase_encoders"):
            for enc_name, encoder in store.doc_graph.phase_encoders.items():
                # Find ENM entries that belong to this encoder's category
                if store.enm:
                    for key in store.enm.keys():
                        if key.type == enc_name:
                            val = store.enm.read_exact(key)
                            if val is None:
                                continue
                            fval = float(val.item() if hasattr(val, "item") else val)
                            angle = encoder.encode(fval)
                            t = ThresholdTruth(
                                entity=key.id,
                                metric=enc_name,
                                value=fval,
                                encoder_name=enc_name,
                                phase_angle=angle,
                            )
                            self._register(t)
                            self._by_encoder.setdefault(enc_name, []).append(
                                t.truth_id)

        elapsed = _time.time() - t0
        print(f"  [TruthStore] Built {len(self._truths)} truth objects "
              f"({self.stats})")

        # Trace truth store build
        if self.tracing and hasattr(self.tracing, "trace_truth_store"):
            counts: dict[str, int] = {}
            for t in self._truths.values():
                counts[t.truth_type] = counts.get(t.truth_type, 0) + 1
            store_name = ""
            if hasattr(store, "store_path"):
                store_name = os.path.basename(store.store_path)
            self.tracing.trace_truth_store(
                store_name=store_name,
                elapsed=elapsed,
                truth_counts=counts,
                total=len(self._truths),
            )

    def _register(self, truth: TruthBase):
        """Register a truth object."""
        self._truths[truth.truth_id] = truth

    # ----- Query methods --------------------------------------------------

    def get(self, truth_id: str) -> TruthBase | None:
        """Get a truth by ID."""
        return self._truths.get(truth_id)

    def get_entity_truths(self, entity: str) -> list[EntityTruth]:
        """Get all entity truths for a named entity."""
        ids = self._by_entity.get(entity, [])
        return [self._truths[tid] for tid in ids if tid in self._truths]

    def get_relation_truth(
        self, head: str, relation: str, tail: str,
    ) -> RelationTruth | None:
        """Get the relation truth for a specific triple."""
        key = f"{head}|{relation}|{tail}"
        ids = self._by_relation.get(key, [])
        if ids:
            return self._truths.get(ids[0])
        return None

    def get_artifact_truth(
        self, category: str, entity_id: str,
    ) -> ArtifactTruth | None:
        """Get the artifact truth for an ENM key."""
        key = f"{category}|{entity_id}"
        tid = self._by_enm_key.get(key)
        if tid:
            return self._truths.get(tid)
        return None

    def get_truth_for_claim(self, claim: "ClaimBase") -> TruthBase | None:
        """Match a claim to its ground-truth object.

        Returns the most specific truth that matches the claim.
        """
        ct = claim.claim_type

        if ct == "fact":
            return self.get_relation_truth(
                claim.head, claim.relation, claim.tail)

        if ct == "artifact":
            return self.get_artifact_truth(
                claim.category, claim.entity_id)

        if ct == "threshold":
            # Try artifact truth first, then threshold
            at = self.get_artifact_truth(claim.metric, claim.entity)
            if at:
                return at
            # Check by encoder
            ids = self._by_encoder.get(claim.metric, [])
            for tid in ids:
                t = self._truths.get(tid)
                if t and hasattr(t, "entity") and t.entity == claim.entity:
                    return t
            return None

        if ct == "rule":
            # Match by first entity
            entities = getattr(claim, "antecedent_entities", [])
            if entities:
                ets = self.get_entity_truths(entities[0])
                if ets:
                    return ets[0]

        return None

    @property
    def stats(self) -> str:
        """Summary statistics."""
        counts: dict[str, int] = {}
        for t in self._truths.values():
            counts[t.truth_type] = counts.get(t.truth_type, 0) + 1
        parts = [f"{k}={v}" for k, v in sorted(counts.items())]
        return ", ".join(parts) if parts else "empty"

    @property
    def all_truths(self) -> list[TruthBase]:
        """All truth objects."""
        return list(self._truths.values())

    # ----- Persistence ----------------------------------------------------

    def save(self, path: str):
        """Save truth store to JSON."""
        data = []
        for t in self._truths.values():
            entry = {
                "truth_id": t.truth_id,
                "truth_type": t.truth_type,
                "doc_id": t.doc_id,
                "section_tag": t.section_tag,
                "confidence": t.confidence,
            }
            # Add type-specific fields
            for k, v in t.__dict__.items():
                if k not in entry:
                    entry[k] = v
            data.append(entry)

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, path: str):
        """Load truth store from JSON."""
        if not os.path.isfile(path):
            return

        with open(path) as f:
            data = json.load(f)

        type_map = {
            "entity": EntityTruth,
            "relation": RelationTruth,
            "threshold": ThresholdTruth,
            "artifact": ArtifactTruth,
        }

        for entry in data:
            tt = entry.get("truth_type", "base")
            cls = type_map.get(tt, TruthBase)
            # Filter to known fields
            known = {f.name for f in cls.__dataclass_fields__.values()}
            filtered = {k: v for k, v in entry.items() if k in known}
            t = cls(**filtered)
            self._register(t)

            # Rebuild indices
            if tt == "entity" and hasattr(t, "entity"):
                self._by_entity.setdefault(t.entity, []).append(t.truth_id)
            elif tt == "relation":
                key = f"{t.head}|{t.relation}|{t.tail}"
                self._by_relation.setdefault(key, []).append(t.truth_id)
            elif tt == "artifact":
                key = f"{t.category}|{t.entity_id}"
                self._by_enm_key[key] = t.truth_id
            elif tt == "threshold" and hasattr(t, "encoder_name"):
                self._by_encoder.setdefault(t.encoder_name, []).append(
                    t.truth_id)
