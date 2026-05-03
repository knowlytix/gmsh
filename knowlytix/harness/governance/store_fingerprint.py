# SPDX-License-Identifier: Apache-2.0
"""Deterministic fingerprint of a GMSExpertStore for bundle binding.

A bundle calibrated on store v1 must not silently serve verdicts
against store v2: the geometry is different, the calibration is
no longer valid, and verdicts become uncharacterized. The
fingerprint is the contract between bundle-time and runtime — both
sides hash the store the same way, and the runtime gate refuses
to bind when hashes diverge.

Components:

  entity_hash    : SHA-256 of the sorted entity vocabulary
  relation_hash  : SHA-256 of the sorted relation vocabulary
  triple_hash    : SHA-256 of the sorted triple set
  model_hash     : SHA-256 of the model state_dict tensor bytes

Each component is reported separately so a mismatch report points
to *what* changed (vocab drift vs weight drift), not just *that*
something changed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore


@dataclass(frozen=True)
class StoreFingerprint:
    """Component-wise digests of a store's identity at calibration time."""

    entity_hash: str
    relation_hash: str
    triple_hash: str
    model_hash: str
    n_entities: int
    n_relations: int
    n_triples: int

    def matches(self, other: "StoreFingerprint") -> bool:
        return (
            self.entity_hash == other.entity_hash
            and self.relation_hash == other.relation_hash
            and self.triple_hash == other.triple_hash
            and self.model_hash == other.model_hash
        )

    def diff(self, other: "StoreFingerprint") -> dict[str, tuple[str, str]]:
        """Return components that differ as ``{component: (mine, other)}``."""
        out: dict[str, tuple[str, str]] = {}
        for field in ("entity_hash", "relation_hash",
                      "triple_hash", "model_hash"):
            mine = getattr(self, field)
            theirs = getattr(other, field)
            if mine != theirs:
                out[field] = (mine[:16], theirs[:16])
        return out

    def to_dict(self) -> dict:
        return asdict(self)


def _hash_strings(items) -> str:
    h = hashlib.sha256()
    for s in sorted(items):
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _hash_triples(triples) -> str:
    h = hashlib.sha256()
    for triple in sorted(triples):
        for term in triple:
            h.update(term.encode("utf-8"))
            h.update(b"\x00")
        h.update(b"\x01")
    return h.hexdigest()


def _hash_model_state(model) -> str:
    """Hash the model's state_dict tensor bytes.

    Bytes are read deterministically from CPU contiguous memory so
    the fingerprint is stable across CPU/GPU and across pickle
    layouts. Returns the hash of an empty state if the model has no
    state_dict (e.g., a stub during testing).
    """
    if model is None or not hasattr(model, "state_dict"):
        return hashlib.sha256(b"no_state_dict").hexdigest()

    h = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state.keys()):
        tensor = state[key]
        h.update(key.encode("utf-8"))
        h.update(b"\x00")
        try:
            arr = tensor.detach().cpu().contiguous().numpy()
        except AttributeError:
            arr = tensor
        h.update(bytes(memoryview(arr)))
        h.update(b"\x01")
    return h.hexdigest()


def compute_store_fingerprint(store: "GMSExpertStore") -> StoreFingerprint:
    """Compute the four digests over a store's identity."""
    entities = list(store.adapter.entity_to_idx.keys())
    relations = list(store.adapter.relation_to_idx.keys())
    triples = (
        list(store.doc_graph.triples) if store.doc_graph is not None else []
    )
    model = getattr(store, "model", None)
    return StoreFingerprint(
        entity_hash=_hash_strings(entities),
        relation_hash=_hash_strings(relations),
        triple_hash=_hash_triples(triples),
        model_hash=_hash_model_state(model),
        n_entities=len(entities),
        n_relations=len(relations),
        n_triples=len(triples),
    )


class StoreFingerprintMismatch(RuntimeError):
    """Runtime store does not match the bundle's calibration-time store."""
