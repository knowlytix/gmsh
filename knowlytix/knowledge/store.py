# SPDX-License-Identifier: Apache-2.0
"""GMSExpertStore: persistent geometric knowledge store.

The core of the expert system. Manages the trained GKG model, ENM register,
compression memory, transport layer, and adapter mappings. Supports persistence,
querying, and runtime growth.
"""

import json
import os
import time

import numpy as np
import torch

from knowlytix.core.graph.gkg import GeometricKnowledgeGraph
from knowlytix.core.graph.transport import RelationalTransport
from knowlytix.core.memory.compression import CompressionMemory
from knowlytix.core.memory.enm import ENMKey, ExactNumericalMemory
from knowlytix.core.memory.router import MemoryRouter
from knowlytix.knowledge.config import DocGMSConfig


class GMSExpertStore:
    """Persistent geometric expert system store.

    Disk layout (store_path/):
      model.pt          - GKG weights
      enm.json          - ENM key-value pairs
      adapter.json      - entity/relation vocabularies
      metadata.json     - ingestion history, stats
      documents/        - ingested markdown copies
    """

    def __init__(self, config: DocGMSConfig, device: torch.device | None = None):
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Core state — populated by ingest or load
        self.model: GeometricKnowledgeGraph | None = None
        self.adapter = None  # GraphToGMS instance
        self.enm: ExactNumericalMemory | None = None
        self.doc_graph = None  # knowlytix.benchmark.graph.DocumentGraph
        self.transport: RelationalTransport | None = None
        self.compression: CompressionMemory | None = None
        self.router: MemoryRouter | None = None
        self.markdown: str = ""

        # Metadata
        self._metadata = {
            "created": None,
            "documents": [],
            "total_ingestions": 0,
        }

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    @property
    def store_path(self) -> str:
        return self.config.store_path

    def assertion_snapshot(self):
        """Load and independently validate the current assertion generation."""
        from pathlib import Path

        from knowlytix.knowledge.assertions.store import load_snapshot
        if not (Path(self.store_path) / 'assertions' / 'CURRENT').exists():
            return None
        return load_snapshot(self.store_path)

    def enforced_assertion_snapshot(self) -> dict | None:
        """The validated snapshot when it is authoritative (enforce mode), else None.

        A shadow generation is only checked at its manifest, so damaged shadow
        artifacts never block the legacy store they accompany. CURRENT is read
        once: the enforced generation it named is the one loaded and validated.
        """
        from knowlytix.knowledge.assertions.store import (
            load_snapshot,
            snapshot_identity,
        )
        identity = snapshot_identity(self.store_path)
        if identity is None or identity[1] != 'enforce':
            return None
        return load_snapshot(self.store_path, generation=identity[0])

    def query_assertions(self, **selection):
        from knowlytix.knowledge.assertions.store import resolve_query
        snapshot = self.assertion_snapshot()
        if snapshot is None:
            raise ValueError('store has no assertion graph')
        return resolve_query(snapshot, **selection)

    def query_calibrated_assertions(self, **selection):
        """Scoped assertion retrieval using the saved, generation-bound GMS gate."""
        from knowlytix.discovery.ontodisc.deployment import query_calibrated_assertions
        return query_calibrated_assertions(self, **selection)

    def exists(self) -> bool:
        """Return whether a saved model file is present at the store path."""
        if os.path.isfile(os.path.join(self.store_path, "model.pt")):
            return True
        return self.enforced_assertion_snapshot() is not None

    def save(self):
        """Save full store state to disk."""
        os.makedirs(self.store_path, exist_ok=True)
        os.makedirs(os.path.join(self.store_path, "documents"), exist_ok=True)

        # Model weights + trained dimensions
        if self.model is not None:
            torch.save(self.model.state_dict(),
                       os.path.join(self.store_path, "model.pt"))
            # Save actual model dimensions (may differ from adapter if expanded)
            model_dims = {
                "num_entities": self.model.num_entities,
                "num_relations": self.model.num_relations,
                # Cap-admissibility metadata for reconstruction at load
                "cap_enabled": bool(getattr(self.model, "cap_enabled", False)),
                "cap_rho_max": float(getattr(self.model, "cap_rho_max", 0.0)),
                "cap_use_diag": bool(getattr(self.model, "cap_use_diag", False)),
                "cap_head_conditioned": bool(getattr(self.model, "cap_head_conditioned", False)),
                "geometry": {k: getattr(self.model.cfg, k) for k in ("d_v", "d_u", "m", "d")},
            }
            with open(os.path.join(self.store_path, "model_dims.json"), "w") as f:
                json.dump(model_dims, f)

        # Adapter mappings
        if self.adapter is not None:
            adapter_data = {
                "entity_to_idx": self.adapter.entity_to_idx,
                "idx_to_entity": {str(k): v for k, v in
                                  self.adapter.idx_to_entity.items()},
                "relation_to_idx": self.adapter.relation_to_idx,
                "idx_to_relation": {str(k): v for k, v in
                                    self.adapter.idx_to_relation.items()},
                "num_entities": self.adapter.num_entities,
                "num_relations": self.adapter.num_relations,
            }
            with open(os.path.join(self.store_path, "adapter.json"), "w") as f:
                json.dump(adapter_data, f, indent=2)

        # ENM entries
        if self.enm is not None:
            enm_data = {}
            for key in self.enm.keys():
                val = self.enm.read_exact(key)
                if val is not None:
                    enm_data[f"{key.type}::{key.id}"] = float(val.item() if hasattr(val, 'item') else val)
            with open(os.path.join(self.store_path, "enm.json"), "w") as f:
                json.dump(enm_data, f, indent=2)

        # Document graph triples
        if self.doc_graph is not None:
            triples_data = [list(t) for t in self.doc_graph.triples]
            with open(os.path.join(self.store_path, "triples.json"), "w") as f:
                json.dump(triples_data, f)

            # Phase encoder ranges
            phase_data = {}
            for name, enc in self.doc_graph.phase_encoders.items():
                phase_data[name] = {"v_min": enc.v_min, "v_max": enc.v_max}
            with open(os.path.join(self.store_path, "phase.json"), "w") as f:
                json.dump(phase_data, f, indent=2)

        # Markdown
        if self.markdown:
            with open(os.path.join(self.store_path, "documents",
                                   "combined.md"), "w") as f:
                f.write(self.markdown)

        # Metadata
        self._metadata["last_saved"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(self.store_path, "metadata.json"), "w") as f:
            json.dump(self._metadata, f, indent=2)

        print(f"  Store saved to {self.store_path}")

    def load(self) -> bool:
        """Load store from disk. Returns False if no existing store."""
        snapshot = self.enforced_assertion_snapshot()
        if not os.path.isfile(os.path.join(self.store_path, 'model.pt')):
            if snapshot is None:
                return False
            self.model = self.adapter = self.enm = self.transport = self.router = None
            self.doc_graph = self.compression = None
            self.markdown = '\n\n'.join(d['text'] for d in snapshot['index']['documents'])
            self._metadata['assertion_generation'] = snapshot['generation']
            self._metadata['assertion_only'] = True
            return True

        if snapshot is not None:
            metadata_path = os.path.join(self.store_path, 'metadata.json')
            with open(metadata_path) as metadata_file:
                model_metadata = json.load(metadata_file)
            if model_metadata.get('assertion_projection_generation') != snapshot['generation']:
                self.model = self.adapter = self.enm = self.transport = self.router = None
                self.doc_graph = self.compression = None
                self.markdown = '\n\n'.join(d['text'] for d in snapshot['index']['documents'])
                self._metadata.update(assertion_model_stale=True, assertion_generation=snapshot['generation'])
                return True

        from knowlytix.benchmark.graph import DocumentGraph
        from knowlytix.core.train_finstructbench import GraphToGMS

        # Adapter
        with open(os.path.join(self.store_path, "adapter.json")) as f:
            ad = json.load(f)

        # Reconstruct doc_graph from saved triples
        self.doc_graph = DocumentGraph()
        triples_path = os.path.join(self.store_path, "triples.json")
        if os.path.isfile(triples_path):
            with open(triples_path) as f:
                for h, r, t in json.load(f):
                    self.doc_graph.add_triple(h, r, t)

        # Phase encoders
        phase_path = os.path.join(self.store_path, "phase.json")
        if os.path.isfile(phase_path):
            with open(phase_path) as f:
                for name, cfg in json.load(f).items():
                    self.doc_graph.add_phase_encoder(name, cfg["v_min"],
                                                     cfg["v_max"])

        # Build adapter from doc_graph, then restore the saved index map: the model's
        # rows follow it. GraphToGMS numbers entities and relations alphabetically,
        # but RuntimeLearner appends new ones at the end, so a store grown at runtime
        # would otherwise load with names pointing at the wrong embedding rows.
        self.adapter = GraphToGMS(self.doc_graph)
        _restore_index(self.adapter, ad)

        # Load model dimensions (may differ from adapter if expanded at runtime)
        dims_path = os.path.join(self.store_path, "model_dims.json")
        cap_enabled = False
        cap_rho_max = 1.5707963267948966
        cap_use_diag = False
        cap_head_conditioned = True
        saved_geometry = None
        if os.path.isfile(dims_path):
            with open(dims_path) as f:
                model_dims = json.load(f)
            model_n_ent = model_dims["num_entities"]
            model_n_rel = model_dims["num_relations"]
            cap_enabled = bool(model_dims.get("cap_enabled", False))
            cap_head_conditioned = bool(model_dims.get("cap_head_conditioned", True))
            saved_geometry = model_dims.get("geometry")
            cap_rho_max = float(model_dims.get("cap_rho_max", cap_rho_max))
            cap_use_diag = bool(model_dims.get("cap_use_diag", False))
        else:
            # Fallback: use adapter dimensions (pre-expansion stores)
            model_n_ent = self.adapter.num_entities
            model_n_rel = self.adapter.num_relations

        # Recreate model with saved dimensions and load weights
        cfg_geo = self.config.geometry
        if saved_geometry:
            from knowlytix.core.config import GeometryConfig
            cfg_geo = GeometryConfig(**saved_geometry)
        self.model = GeometricKnowledgeGraph(
            num_entities=model_n_ent,
            num_relations=model_n_rel,
            cfg=cfg_geo,
            cap_enabled=cap_enabled,
            cap_rho_max=cap_rho_max,
            cap_use_diag=cap_use_diag,
            cap_head_conditioned=cap_head_conditioned,
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(os.path.join(self.store_path, "model.pt"),
                       map_location=self.device, weights_only=True)
        )
        self.model.eval()

        # ENM
        enm_path = os.path.join(self.store_path, "enm.json")
        if os.path.isfile(enm_path):
            with open(enm_path) as f:
                enm_data = json.load(f)
            self.enm = ExactNumericalMemory(
                capacity=max(len(enm_data) + 100, 1000)
            )
            for composite_key, val in enm_data.items():
                parts = composite_key.split("::", 1)
                key = ENMKey(type=parts[0], id=parts[1])
                self.enm.write(key, np.array(val))
                # Also write to doc_graph ENM
                self.doc_graph.store_value(parts[0], parts[1], val)
        else:
            self.enm = ExactNumericalMemory(capacity=1000)

        # Transport
        self.transport = RelationalTransport(self.model)

        # Compression + Router
        m = cfg_geo.m
        self.compression = CompressionMemory(k=self.config.memory.k, d=m)
        self.router = MemoryRouter(self.enm, self.compression)

        # Markdown
        md_path = os.path.join(self.store_path, "documents", "combined.md")
        if os.path.isfile(md_path):
            with open(md_path) as f:
                self.markdown = f.read()

        # Metadata
        meta_path = os.path.join(self.store_path, "metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                self._metadata = json.load(f)

        print(f"  Store loaded from {self.store_path}")
        self._print_stats()
        return True

    # -----------------------------------------------------------------
    # GMS Operations
    # -----------------------------------------------------------------

    @staticmethod
    def _canon_entity(name: str) -> str:
        # Mirror DocumentGraph._canon_entity — entity_to_idx keys are stored
        # in canonical (lower, single-spaced) form after the ingest fix, so
        # API callers that pass raw casing ("MDL-001", "Credit Decisioning")
        # need to be normalised here on the read path too.
        return " ".join(name.split()).lower()

    def score_triple(self, head: str, rel: str, tail: str) -> float | None:
        """Score plausibility of (head, rel, tail). Lower = more plausible.

        For cap-trained stores (``model.cap_enabled=True``) this returns
        the geodesic distance to the *conditioned* cap center
        ``c_{h,r}=normalize(Lambda_r * R_r v_h)`` rather than the raw
        rotor output — so admissibility comparisons against the per-
        relation radius ``rho_r`` are consistent with training.
        """
        if self.model is None or self.adapter is None:
            return None
        mode = self._metadata.get('assertion_projection_scoring')
        if mode == 'canonical_triple_v1':
            from knowlytix.core.ontodisc.kg.canonicalize import canonical_triple
            head, rel, tail = canonical_triple(head, rel, tail)
        elif mode is not None:
            raise ValueError(f'Unknown assertion projection scoring mode: {mode}')
        h_idx = self.adapter.entity_to_idx.get(self._canon_entity(head))
        t_idx = self.adapter.entity_to_idx.get(self._canon_entity(tail))
        r_idx = self.adapter.relation_to_idx.get(rel)
        if h_idx is None or t_idx is None or r_idx is None:
            return None
        # Guard against indices beyond model's current embedding size
        if (h_idx >= self.model.num_entities or
                t_idx >= self.model.num_entities or
                r_idx >= self.model.num_relations):
            return None
        with torch.no_grad():
            h_t = torch.tensor([h_idx], device=self.device)
            r_t = torch.tensor([r_idx], device=self.device)
            t_t = torch.tensor([t_idx], device=self.device)
            if getattr(self.model, "cap_enabled", False):
                score = self.model.cap_distance(h_t, r_t, t_t)
            else:
                score = self.model.score_triple(h_t, r_t, t_t)
        return score.item()

    def score_triple_emb(self, head: str, rel: str, tail_vec) -> "float | None":
        """Cap distance of ``(head, rel, <tail embedding>)``. Lower = more plausible.

        Like :meth:`score_triple`, but the tail is supplied as a ``d_v``-space
        embedding rather than a trained-entity name, so a value the graph never
        stored as an entity (an out-of-vocabulary polarity token such as
        ``optional``) can still be scored against the relation cap. The embedding
        must come from the same v-encoder that warm-started the entity table
        (``FineTunedEmbedding.encode`` of the saved ``tuned_encoder``, whose
        ``out_dim`` equals ``d_v``); it is projected through the shared Stiefel
        ``P_v`` exactly as :meth:`DualEmbedding.project_v` projects a stored row,
        so it lands on the same sphere the operators and caps act on. The head and
        relation are still looked up by index (they are in vocabulary). Requires a
        cap-trained store; returns ``None`` otherwise or on an unknown head/relation
        or a dimension mismatch.
        """
        import torch
        import torch.nn.functional as F

        if self.model is None or self.adapter is None:
            return None
        if not getattr(self.model, "cap_enabled", False):
            return None
        h_idx = self.adapter.entity_to_idx.get(self._canon_entity(head))
        r_idx = self.adapter.relation_to_idx.get(rel)
        if h_idx is None or r_idx is None:
            return None
        if h_idx >= self.model.num_entities or r_idx >= self.model.num_relations:
            return None
        de = self.model.dual_emb
        vec = torch.as_tensor(tail_vec, dtype=torch.float32,
                              device=self.device).reshape(-1)
        if vec.shape[0] != de.d_v:
            return None
        with torch.no_grad():
            h_t = torch.tensor([h_idx], device=self.device)
            r_t = torch.tensor([r_idx], device=self.device)
            c = self.model.cap_center(h_t, r_t)                  # (1, m)
            v_t = F.normalize(vec @ de.P_v.T, dim=-1).unsqueeze(0)  # (1, m)
            dot = (c * v_t).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            return float(torch.arccos(dot)[0])

    def cap_radius(self, rel: str, head: str | None = None) -> float | None:
        """Learned admissible radius ρ_r for this relation; None if not cap-trained."""
        if self.model is None or self.adapter is None:
            return None
        if not getattr(self.model, "cap_enabled", False):
            return None
        r_idx = self.adapter.relation_to_idx.get(rel)
        if r_idx is None or r_idx >= self.model.num_relations:
            return None
        with torch.no_grad():
            if head is not None:
                h_idx = self.adapter.entity_to_idx.get(self._canon_entity(head))
                if h_idx is None or h_idx >= self.model.num_entities:
                    return None
                return float(self.model.cap_radius_hr(
                    torch.tensor([h_idx], device=self.device),
                    torch.tensor([r_idx], device=self.device)).item())
            return float(self.model.cap_radii()[r_idx].item())

    def cap_radii_all(self) -> dict[str, float] | None:
        """All per-relation admissible radii; None if not cap-trained."""
        if self.model is None or self.adapter is None:
            return None
        if not getattr(self.model, "cap_enabled", False):
            return None
        with torch.no_grad():
            radii = self.model.cap_radii().cpu().tolist()
        return {
            name: float(radii[idx])
            for name, idx in self.adapter.relation_to_idx.items()
            if idx < len(radii)
        }

    def tension_energy(self, entity_a: str, entity_b: str) -> float | None:
        """Tension energy between two entities in u-space.

        Larger values indicate stronger logical opposition. The absolute scale
        is a property of the trained store, not a fixed constant: the cut points
        that separate agreement, irrelevance and contradiction must be obtained
        by calibration on labeled pairs (as in the per-relation cutoffs of
        ``core.graph.admissibility``), never assumed from theoretical values."""
        if self.model is None or self.adapter is None:
            return None
        a_idx = self.adapter.entity_to_idx.get(self._canon_entity(entity_a))
        b_idx = self.adapter.entity_to_idx.get(self._canon_entity(entity_b))
        if a_idx is None or b_idx is None:
            return None
        if a_idx >= self.model.num_entities or b_idx >= self.model.num_entities:
            return None
        with torch.no_grad():
            te = self.model.tension_energy_pairs(
                torch.tensor([a_idx], device=self.device),
                torch.tensor([b_idx], device=self.device),
            )
        return te.item()

    def check_holonomy(self, relation_path: list[str],
                        direct_relation: str) -> float | None:
        """Holonomy defect for path vs direct edge. 0=consistent."""
        if self.transport is None or self.adapter is None:
            return None
        path_idx = [self.adapter.relation_to_idx.get(r)
                     for r in relation_path]
        direct_idx = self.adapter.relation_to_idx.get(direct_relation)
        if None in path_idx or direct_idx is None:
            return None
        # Guard against indices beyond model's rotor capacity
        if any(i >= self.model.num_relations for i in path_idx):
            return None
        if direct_idx >= self.model.num_relations:
            return None
        with torch.no_grad():
            defect = self.transport.path_holonomy(path_idx, direct_idx)
        return defect.item()

    def is_path_consistent(self, relation_path: list[str],
                            direct_relation: str) -> bool | None:
        """True if holonomy defect <= tau_path."""
        defect = self.check_holonomy(relation_path, direct_relation)
        if defect is None:
            return None
        return defect <= self.config.verify.tau_path

    def lookup_enm(self, category: str, entity_id: str) -> float | None:
        """Exact numeric lookup with SHA-256 integrity.

        Searches by (type, id) since ENMKey hashes include timestamp.
        """
        snapshot = self.enforced_assertion_snapshot()
        if snapshot is not None:
            raise ValueError('use query_assertions(numeric=True) for scope-aware numeric lookup')
        if self.enm is None:
            return None
        for key in self.enm.keys():
            if key.type == category and key.id == entity_id:
                val = self.enm.read_exact(key)
                if val is not None:
                    return float(val.item() if hasattr(val, 'item') else val)
        return None

    def query_exact(self, category: str, entity_id: str):
        """Compat adapter for benchmark generators that expect a tensor with
        ``.item()``. Wraps :meth:`lookup_enm` (which returns a float)."""
        v = self.lookup_enm(category, entity_id)
        return None if v is None else torch.tensor(float(v))

    def query_phase(self, encoder: str, value: float, threshold: float, op: str):
        """Compat threshold comparison for benchmark generators. Returns
        ``(satisfied, margin)``; ``op`` is ``'ge'`` or ``'lt'``. The encoder name
        is accepted for signature compatibility; the test is a plain numeric
        comparison against the exact value."""
        if value is None:
            return (None, None)
        ok = (value >= threshold) if op == "ge" else (value < threshold)
        return (bool(ok), float(value - threshold))

    def enm_by_type(self, category: str) -> list[tuple[str, float]]:
        """All scalar ENM entries ``(id, value)`` of a given type."""
        out: list[tuple[str, float]] = []
        if self.enm is None:
            return out
        for key in self.enm.keys():
            if key.type != category:
                continue
            val = self.enm.read_exact(key)
            if val is None:
                continue
            if getattr(val, "size", 1) != 1:  # scalars only
                continue
            out.append((key.id, float(val.item() if hasattr(val, "item")
                                      else val)))
        return out

    def _u_dim(self) -> int:
        """Ambient u-space dimension of the trained model (falls back to 32)."""
        try:
            return int(self.model.dual_emb.P_u.shape[0])
        except (AttributeError, IndexError, TypeError):
            return 32

    def enm_rank(self, category: str, which: str = "highest"
                 ) -> list[tuple[str, float]]:
        """Entities of ``category`` ranked by value through u-space logic.

        Ordering is the directed-entailment decision of
        :class:`~knowlytix.core.geometry.numeric_order.NumericUSpaceOrder` over
        the exact ENM values -- the same operator the GMS uses for logical
        entailment, so the result is verifiable on the same footing.
        """
        from knowlytix.core.geometry.numeric_order import NumericUSpaceOrder
        items = self.enm_by_type(category)
        if not items:
            return []
        return NumericUSpaceOrder(items, dim=self._u_dim()).rank(which)

    def enm_extreme(self, category: str, which: str = "highest"):
        """The argmax/argmin entity of ``category`` with its u-space margin.

        Returns an
        :class:`~knowlytix.core.geometry.numeric_order.OrderResult`
        (``entity``, ``value``, ``margin``, ``relevance``) or ``None``.
        """
        from knowlytix.core.geometry.numeric_order import NumericUSpaceOrder
        items = self.enm_by_type(category)
        if not items:
            return None
        return NumericUSpaceOrder(items, dim=self._u_dim()).extreme(which)

    def enm_compare(self, category: str, id_a: str, id_b: str):
        """Compare two entities of ``category``; the higher one + its margin.

        Returns an
        :class:`~knowlytix.core.geometry.numeric_order.OrderResult` or ``None``.
        """
        from knowlytix.core.geometry.numeric_order import NumericUSpaceOrder
        items = self.enm_by_type(category)
        if not items:
            return None
        return NumericUSpaceOrder(items, dim=self._u_dim()).compare(id_a, id_b)

    def link_predict(self, head: str, relation: str,
                      top_k: int = 10) -> list[tuple[str, float]]:
        """Top-k tail predictions for (head, relation, ?).

        Uses type-constrained filtered evaluation:
          1. Self-entity (head) is excluded — (h, r, h) is never meaningful.
          2. Only entities that appear as tails for this relation in the
             knowledge graph are scored.  This prevents numeric entities
             from polluting categorical predictions and vice versa.
        """
        if self.model is None or self.adapter is None:
            return []
        h_idx = self.adapter.entity_to_idx.get(self._canon_entity(head))
        r_idx = self.adapter.relation_to_idx.get(relation)
        if h_idx is None or r_idx is None:
            return []
        r_idx_int = self.adapter.relation_to_idx[relation]
        with torch.no_grad():
            scores = self.model.score_all_tails(
                torch.tensor([h_idx], device=self.device),
                torch.tensor([r_idx], device=self.device),
            ).squeeze(0)
            # Filtered setting: mask self-entity
            scores[h_idx] = float('inf')
            # Type constraint: mask entities not seen as tails for this relation
            valid_tails = self.adapter.tail_per_relation.get(r_idx_int, set())
            if valid_tails:
                mask = torch.ones(len(scores), dtype=torch.bool,
                                  device=self.device)
                for t_idx in valid_tails:
                    mask[t_idx] = False
                scores[mask] = float('inf')
            top_vals, top_idx = scores.topk(min(top_k, len(scores)),
                                            largest=False)
        results = []
        for i in range(len(top_idx)):
            if top_vals[i].item() == float('inf'):
                break
            ent = self.adapter.idx_to_entity.get(top_idx[i].item(), "?")
            results.append((ent, top_vals[i].item()))
        return results

    def query_triples(self, head=None, relation=None, tail=None):
        """Pattern-match current scoped projections for enforced assertion stores."""
        snapshot = self.enforced_assertion_snapshot()
        if snapshot is not None:
            return [tuple(row['triple']) for row in snapshot['projections']['triples']
                    if all(wanted is None or wanted == value
                           for wanted, value in zip((head, relation, tail), row['triple']))]
        if self.doc_graph is None:
            return []
        return self.doc_graph.query_triples(head=head, relation=relation,
                                            tail=tail)

    def find_contradictions(self):
        """Find pass/fail contradictions in document graph."""
        if self.doc_graph is None:
            return []
        return self.doc_graph.find_contradictions()

    def entity_exists(self, name: str) -> bool:
        """Return whether the canonicalized entity ``name`` is in the adapter vocabulary."""
        return (
            self.adapter is not None
            and self._canon_entity(name) in self.adapter.entity_to_idx
        )

    def relation_exists(self, name: str) -> bool:
        """Return whether relation ``name`` is in the adapter vocabulary."""
        return (self.adapter is not None
                and name in self.adapter.relation_to_idx)

    def fuzzy_match_entity(self, name: str) -> str | None:
        """Find a unique entity name by normalized string matching.

        Matching levels, in order of precedence:

          1. Exact match (as stored) --- always wins.
          2. Case-insensitive exact --- returned only if unique.
          3. Bidirectional substring --- returned only if unique.

        When two or more entities tie at a given level we refuse
        rather than pick arbitrarily.  The previous version returned
        the first substring hit in iteration order, which produced
        silent mis-resolutions (e.g. ``"RandomForest"`` resolving to
        ``"DecisionTree, RandomForest, GradientBoosting"`` when the
        bare name was not yet in the vocabulary).  Silent
        mis-resolution propagated into the write-time contradiction
        gate and caused false-positive rejections --- hence the
        bank-grade rule: ambiguous input must return ``None`` so
        the caller knows the match was not well-defined.
        """
        if self.adapter is None:
            return None
        # Exact
        if name in self.adapter.entity_to_idx:
            return name
        name_lower = name.strip().lower()
        # Case-insensitive exact (unique)
        case_hits = [
            ent for ent in self.adapter.entity_to_idx
            if ent.lower() == name_lower
        ]
        if len(case_hits) == 1:
            return case_hits[0]
        if len(case_hits) > 1:
            return None
        # Substring (unique)
        sub_hits = [
            ent for ent in self.adapter.entity_to_idx
            if name_lower in ent.lower() or ent.lower() in name_lower
        ]
        if len(sub_hits) == 1:
            return sub_hits[0]
        return None

    # -----------------------------------------------------------------
    # Convenience accessors
    # -----------------------------------------------------------------

    @property
    def triples(self):
        """Shortcut for doc_graph.triples (used by eval generators)."""
        if self.doc_graph is None:
            return []
        return self.doc_graph.triples

    # -----------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------

    def stats(self) -> dict:
        """Return counts of entities, relations, triples, ENM entries and documents.

        Flags an adapter/model mismatch by adding model counts and unembedded totals
        when the model holds fewer entities or relations than the adapter.
        """
        n_adapter_ent = self.adapter.num_entities if self.adapter else 0
        n_adapter_rel = self.adapter.num_relations if self.adapter else 0
        n_model_ent = self.model.num_entities if self.model else 0
        n_model_rel = self.model.num_relations if self.model else 0
        result = {
            "entities": n_adapter_ent,
            "relations": n_adapter_rel,
            "triples": len(self.doc_graph.triples) if self.doc_graph else 0,
            "enm_entries": self.enm.size() if self.enm else 0,
            "documents": len(self._metadata.get("documents", [])),
            "store_path": self.store_path,
        }
        # Flag adapter/model mismatch (entities added but not yet embedded)
        if n_model_ent < n_adapter_ent or n_model_rel < n_adapter_rel:
            result["model_entities"] = n_model_ent
            result["model_relations"] = n_model_rel
            result["unembedded_entities"] = n_adapter_ent - n_model_ent
            result["unembedded_relations"] = n_adapter_rel - n_model_rel
        return result

    def _print_stats(self):
        s = self.stats()
        print(f"  Entities:  {s['entities']}")
        print(f"  Relations: {s['relations']}")
        print(f"  Triples:   {s['triples']}")
        print(f"  ENM:       {s['enm_entries']}")
        print(f"  Documents: {s['documents']}")
        if "unembedded_entities" in s:
            print(f"  WARNING: {s['unembedded_entities']} entities / "
                  f"{s['unembedded_relations']} relations lack embeddings "
                  f"(model has {s['model_entities']}/{s['model_relations']})")


def _restore_index(adapter, saved: dict) -> bool:
    """Apply a saved ``adapter.json`` index map to a freshly built adapter.

    Only when it covers exactly the same entities and relations (otherwise the
    rebuilt alphabetical map is kept). The triple index tensors and per-relation
    tail sets are rebuilt from the restored map. Returns whether it was applied.
    """
    import torch

    ent = {str(k): int(v) for k, v in (saved.get("entity_to_idx") or {}).items()}
    rel = {str(k): int(v) for k, v in (saved.get("relation_to_idx") or {}).items()}
    if (set(ent) != set(adapter.entity_to_idx) or set(rel) != set(adapter.relation_to_idx)
            or sorted(ent.values()) != list(range(len(ent)))
            or sorted(rel.values()) != list(range(len(rel)))):
        return False
    if ent == adapter.entity_to_idx and rel == adapter.relation_to_idx:
        return True
    # remap each triple tensor: old (alphabetical) index -> saved index
    old_ent = {i: e for e, i in adapter.entity_to_idx.items()}
    old_rel = {i: r for r, i in adapter.relation_to_idx.items()}
    adapter.entity_to_idx, adapter.relation_to_idx = ent, rel
    adapter.idx_to_entity = {i: e for e, i in ent.items()}
    adapter.idx_to_relation = {i: r for r, i in rel.items()}
    adapter.heads = torch.tensor([ent[old_ent[int(i)]] for i in adapter.heads], dtype=torch.long)
    adapter.tails = torch.tensor([ent[old_ent[int(i)]] for i in adapter.tails], dtype=torch.long)
    adapter.relations = torch.tensor([rel[old_rel[int(i)]] for i in adapter.relations],
                                     dtype=torch.long)
    adapter.tail_per_relation = {}
    for r, t in zip(adapter.relations.tolist(), adapter.tails.tolist()):
        adapter.tail_per_relation.setdefault(r, set()).add(t)
    return True
