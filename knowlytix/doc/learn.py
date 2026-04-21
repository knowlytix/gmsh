# SPDX-License-Identifier: Apache-2.0
"""Runtime memory growth via Riemannian gradient descent.

Two-phase protocol:
  Phase 1 (synchronous, fast): Verify triples → add to graph → expand adapter
  Phase 2 (background, async): Expand embeddings + rotors → Riemannian GD

Phase 1 runs inline so queries against the graph work immediately.
Phase 2 runs in a background thread so geometric training doesn't block
interactive use.
"""

import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from knowlytix.core.geometry.cayley import CayleyRotor
from knowlytix.core.geometry.sphere import normalize, geodesic_distance
from knowlytix.core.graph.encoders import encode_texts
from knowlytix.core.memory.enm import ENMKey

from knowlytix.doc.config import LearnConfig
from knowlytix.doc.extract import ExtractedTriple
from knowlytix.doc.store import GMSExpertStore


@dataclass
class LearnResult:
    """Result of a runtime memory write."""
    accepted: bool = False
    reason: str = ""
    new_entities: list[str] = field(default_factory=list)
    new_relations: list[str] = field(default_factory=list)
    n_triples_added: int = 0
    training_queued: bool = False       # True if background training was queued
    scores_before: list[float] = field(default_factory=list)
    scores_after: list[float] = field(default_factory=list)
    n_steps_run: int = 0


@dataclass
class _TrainJob:
    """A queued background training job."""
    triple_indices: list[tuple]
    n_existing_ent: int
    n_existing_rel: int
    n_new_ent: int
    n_new_rel: int
    new_entity_names: list[str]
    device: torch.device
    timestamp: float = 0.0


class RuntimeLearner:
    """Grow GMS with verified knowledge. Training runs in background.

    Protocol:
    Phase 1 — synchronous (fast):
      1. Check contradictions BEFORE accepting (tension energy gate)
      2. Add verified triples to doc_graph (instant graph queries)
      3. Expand adapter indices (entity_to_idx, relation_to_idx)
      4. Write numeric values to ENM
      5. Queue training job

    Phase 2 — background thread:
      6. Expand embedding tables (sentence-transformer init)
      7. Add new Cayley rotors for new relations
      8. Freeze existing params
      9. Riemannian GD on new embeddings + new rotors
     10. Unfreeze, update model counts
     11. Save store
    """

    def __init__(self, store: GMSExpertStore, config: LearnConfig):
        self.store = store
        self.config = config
        self._train_queue: list[_TrainJob] = []
        self._train_thread: threading.Thread | None = None
        self._train_lock = threading.Lock()
        self.is_training = False
        self.last_train_result: str = ""

    # =================================================================
    # Phase 1: Synchronous — verify, add to graph, queue training
    # =================================================================

    def learn_triples(self, new_triples: list[ExtractedTriple],
                       device: torch.device) -> LearnResult:
        """Verify and accept triples. Training runs in background."""
        model = self.store.model
        adapter = self.store.adapter
        if model is None or adapter is None:
            return LearnResult(accepted=False, reason="Store not initialized")

        # ---- Identify what's new ----
        existing_ents = set(adapter.entity_to_idx.keys())
        existing_rels = set(adapter.relation_to_idx.keys())

        new_entity_names = set()
        new_relation_names = set()
        for t in new_triples:
            if t.head not in existing_ents:
                new_entity_names.add(t.head)
            if t.tail not in existing_ents:
                new_entity_names.add(t.tail)
            rel = t.relation.strip().lower().replace(" ", "_")
            if rel not in existing_rels:
                new_relation_names.add(rel)

        new_entity_names = sorted(new_entity_names)
        new_relation_names = sorted(new_relation_names)

        result = LearnResult(
            new_entities=list(new_entity_names),
            new_relations=list(new_relation_names),
        )

        # ---- Pre-write contradiction check ----
        if self.config.contradiction_gate:
            rejected = self._check_contradictions_before(new_triples)
            if rejected:
                result.accepted = False
                result.reason = f"Contradiction gate: {rejected}"
                return result

        # ---- Add triples to doc_graph (instant for graph queries) ----
        for t in new_triples:
            self.store.doc_graph.add_triple(t.head, t.relation, t.tail)
        result.n_triples_added = len(new_triples)

        # ---- Expand adapter indices (preserve existing) ----
        for rel_name in new_relation_names:
            new_idx = adapter.num_relations
            adapter.relation_to_idx[rel_name] = new_idx
            adapter.idx_to_relation[new_idx] = rel_name
            adapter.num_relations = new_idx + 1

        for ent_name in new_entity_names:
            new_idx = adapter.num_entities
            adapter.entity_to_idx[ent_name] = new_idx
            adapter.idx_to_entity[new_idx] = ent_name
            adapter.num_entities = new_idx + 1

        # ---- Write numeric values to ENM ----
        for t in new_triples:
            if t.numeric_value is not None and self.store.enm is not None:
                rel = t.relation.strip().lower().replace(" ", "_")
                enm_key = ENMKey(type=rel, id=f"{t.head}/{t.tail}")
                self.store.enm.write(enm_key, np.array(t.numeric_value))

        # ---- Queue background training if there are new entities/relations ----
        if new_entity_names or new_relation_names:
            triple_indices = self._resolve_triples(new_triples, adapter)
            if triple_indices:
                job = _TrainJob(
                    triple_indices=triple_indices,
                    n_existing_ent=adapter.num_entities - len(new_entity_names),
                    n_existing_rel=adapter.num_relations - len(new_relation_names),
                    n_new_ent=len(new_entity_names),
                    n_new_rel=len(new_relation_names),
                    new_entity_names=list(new_entity_names),
                    device=device,
                    timestamp=time.time(),
                )
                self._queue_training(job)
                result.training_queued = True

        result.accepted = True
        n_ent = len(new_entity_names)
        n_rel = len(new_relation_names)
        parts = [f"{result.n_triples_added} triples added to graph"]
        if n_ent:
            parts.append(f"{n_ent} new entities")
        if n_rel:
            parts.append(f"{n_rel} new relations")
        if result.training_queued:
            parts.append("training queued in background")
        result.reason = ". ".join(parts)
        return result

    def learn_enm(self, category: str, entity_id: str, value: float) -> bool:
        """Write exact numeric value to ENM."""
        if self.store.enm is None:
            return False
        key = ENMKey(type=category, id=entity_id)
        self.store.enm.write(key, np.array(value))
        self.store.doc_graph.store_value(category, entity_id, value)
        return True

    # =================================================================
    # Pre-write contradiction check
    # =================================================================

    def _check_contradictions_before(self, new_triples: list[ExtractedTriple]) -> str:
        """Check for contradictions BEFORE adding triples.

        Returns empty string if OK, or description of contradiction.
        """
        from knowlytix.doc.verify import check_contradictions
        from knowlytix.doc.config import VerifyConfig

        # Build entity pairs: new triple entities vs existing connected entities
        pairs = []
        for t in new_triples:
            h_match = self.store.fuzzy_match_entity(t.head)
            t_match = self.store.fuzzy_match_entity(t.tail)

            # Check if the new triple's entities contradict existing ones
            if h_match and t_match and h_match != t.head and t_match != t.tail:
                # Both ends match existing entities — check tension
                pairs.append((h_match, t_match))

            # Also check each matched entity against its existing neighbors
            if h_match:
                related = self.store.query_triples(head=h_match)
                for _, _, existing_t in related[:5]:
                    matched = self.store.fuzzy_match_entity(existing_t)
                    if matched and t_match and matched != t_match:
                        pairs.append((t_match, matched))

        if not pairs:
            return ""

        pairs = list(set(pairs))
        vconfig = VerifyConfig(tau_contra=self.store.config.verify.tau_contra)
        contras = check_contradictions(pairs, self.store, vconfig)
        n_contras = sum(1 for c in contras if c.is_contradiction)

        if n_contras > 0:
            details = []
            for c in contras:
                if c.is_contradiction:
                    details.append(
                        f"{c.entity_a} <-> {c.entity_b} "
                        f"(tension={c.tension_energy:.3f})"
                    )
            return f"{n_contras} contradictions: " + "; ".join(details[:3])
        return ""

    # =================================================================
    # Phase 2: Background training thread
    # =================================================================

    def _queue_training(self, job: _TrainJob):
        """Add a training job to the queue. Call flush_training() to start."""
        with self._train_lock:
            self._train_queue.append(job)

    def flush_training(self):
        """Start background training on all queued jobs. Call after all
        Phase 1 writes are done (e.g., after all sources are merged)."""
        with self._train_lock:
            if not self._train_queue:
                return
        self._start_training_thread()

    def _start_training_thread(self):
        """Start background training thread if not already running."""
        if self._train_thread is not None and self._train_thread.is_alive():
            return  # Already running, it will pick up the queued job
        self._train_thread = threading.Thread(
            target=self._train_loop, daemon=True,
            name="gms-background-train",
        )
        self._train_thread.start()

    def _train_loop(self):
        """Process training jobs from the queue — batch all pending jobs."""
        while True:
            with self._train_lock:
                if not self._train_queue:
                    self.is_training = False
                    return
                # Drain all queued jobs into one batch
                jobs = list(self._train_queue)
                self._train_queue.clear()

            self.is_training = True
            # Merge all triple indices from all jobs
            all_triple_indices = []
            all_new_entity_names = []
            for job in jobs:
                all_triple_indices.extend(job.triple_indices)
                all_new_entity_names.extend(job.new_entity_names)
            # Deduplicate entity names
            all_new_entity_names = sorted(set(all_new_entity_names))

            merged_job = _TrainJob(
                triple_indices=all_triple_indices,
                n_existing_ent=jobs[0].n_existing_ent,
                n_existing_rel=jobs[0].n_existing_rel,
                n_new_ent=len(all_new_entity_names),
                n_new_rel=sum(j.n_new_rel for j in jobs),
                new_entity_names=all_new_entity_names,
                device=jobs[0].device,
            )
            try:
                self._run_training_job(merged_job)
            except Exception as e:
                self.last_train_result = f"Training error: {e}"
                print(f"  [BackgroundTrain] Error: {e}")

    def _run_training_job(self, job: _TrainJob):
        """Execute a training job: expand model + Riemannian GD.

        Uses CURRENT model state (not stale job metadata) to determine
        what needs expanding, since Phase 1 may have queued multiple jobs
        while Phase 2 hadn't started yet.
        """
        model = self.store.model
        adapter = self.store.adapter
        device = job.device
        t0 = time.time()

        # What the model currently has vs what the adapter needs
        n_model_ent = model.num_entities
        n_model_rel = model.num_relations
        n_adapter_ent = adapter.num_entities
        n_adapter_rel = adapter.num_relations

        n_new_ent = n_adapter_ent - n_model_ent
        n_new_rel = n_adapter_rel - n_model_rel

        if n_new_ent <= 0 and n_new_rel <= 0:
            # Model already covers everything — skip
            return

        # Find names of entities that need embeddings
        new_names = [
            name for name, idx in adapter.entity_to_idx.items()
            if idx >= n_model_ent
        ]

        # Filter triple indices to ones the expanded model can handle
        target_ent = n_model_ent + max(n_new_ent, 0)
        target_rel = n_model_rel + max(n_new_rel, 0)
        valid_triples = [
            (h, r, t) for h, r, t in job.triple_indices
            if h < target_ent and t < target_ent and r < target_rel
        ]

        print(f"  [BackgroundTrain] Starting: {n_new_ent} entities, "
              f"{n_new_rel} relations, {len(valid_triples)} triples")

        # 1. Expand embeddings for new entities
        if n_new_ent > 0 and new_names:
            self._expand_embeddings(new_names, device)

        # 2. Add new Cayley rotors for new relations
        if n_new_rel > 0:
            self._expand_rotors(n_new_rel, device)

        # 3. Incremental training: freeze existing, train new
        if valid_triples:
            n_steps = self._incremental_train(
                valid_triples,
                n_model_ent, n_model_rel,
                max(n_new_ent, 0), max(n_new_rel, 0),
                device,
            )
        else:
            n_steps = 0

        # 4. Sync model counts with adapter
        model.num_entities = adapter.num_entities
        model.num_relations = adapter.num_relations

        elapsed = time.time() - t0
        self.last_train_result = (
            f"Trained: {job.n_new_ent} entities, {job.n_new_rel} relations, "
            f"{n_steps} steps in {elapsed:.1f}s"
        )
        print(f"  [BackgroundTrain] Done: {self.last_train_result}")

        # 5. Auto-save
        try:
            self.store.save()
            print("  [BackgroundTrain] Store saved")
        except Exception as e:
            print(f"  [BackgroundTrain] Save error: {e}")

    def wait_for_training(self, timeout: float = 300.0) -> bool:
        """Wait for background training to finish. Returns True if completed."""
        if self._train_thread is None:
            return True
        self._train_thread.join(timeout=timeout)
        return not self._train_thread.is_alive()

    def training_status(self) -> str:
        """Return current training status."""
        with self._train_lock:
            queued = len(self._train_queue)
        if self.is_training:
            return f"Training in progress ({queued} queued)"
        if self.last_train_result:
            return f"Last: {self.last_train_result}"
        return "Idle"

    # =================================================================
    # Model expansion helpers
    # =================================================================

    def _expand_embeddings(self, new_names: list[str], device: torch.device):
        """Expand GKG embedding tables with new entities."""
        model = self.store.model
        n_existing = model.num_entities
        n_new = len(new_names)
        n_total = n_existing + n_new
        d_v = model.dual_emb.d_v
        d_u = model.dual_emb.d_u

        # Encode new entities via sentence-transformers
        try:
            new_v_raw = encode_texts(
                new_names, "sentence-transformers/all-MiniLM-L6-v2",
                str(device),
            )
            new_u_raw = encode_texts(
                new_names, "sentence-transformers/nli-mpnet-base-v2",
                str(device),
            )
        except Exception:
            new_v_raw = torch.randn(n_new, 384, device=device)
            new_u_raw = torch.randn(n_new, 384, device=device)

        # Match dimensions
        if new_v_raw.shape[1] != d_v:
            new_v_raw = new_v_raw[:, :d_v] if new_v_raw.shape[1] > d_v else \
                torch.nn.functional.pad(new_v_raw, (0, d_v - new_v_raw.shape[1]))
        if new_u_raw.shape[1] != d_u:
            new_u_raw = new_u_raw[:, :d_u] if new_u_raw.shape[1] > d_u else \
                torch.nn.functional.pad(new_u_raw, (0, d_u - new_u_raw.shape[1]))

        # Create expanded embeddings
        new_v_embed = nn.Embedding(n_total, d_v).to(device)
        new_u_embed = nn.Embedding(n_total, d_u).to(device)
        new_spec = nn.Parameter(torch.zeros(n_total, device=device))
        new_conf = nn.Parameter(torch.full((n_total,), 2.0, device=device))

        # Copy existing
        new_v_embed.weight.data[:n_existing] = model.dual_emb.v_embed.weight.data
        new_u_embed.weight.data[:n_existing] = model.dual_emb.u_embed.weight.data
        new_spec.data[:n_existing] = model.dual_emb.specificity_logit.data
        new_conf.data[:n_existing] = model.dual_emb.confidence_logit.data

        # Init new from pretrained encoders
        new_v_embed.weight.data[n_existing:] = new_v_raw.to(device)
        new_u_embed.weight.data[n_existing:] = new_u_raw.to(device)

        # Swap into model
        model.dual_emb.v_embed = new_v_embed
        model.dual_emb.u_embed = new_u_embed
        model.dual_emb.specificity_logit = new_spec
        model.dual_emb.confidence_logit = new_conf
        model.dual_emb.num_entities = n_total
        model.num_entities = n_total

    def _expand_rotors(self, n_new_rels: int, device: torch.device):
        """Add new Cayley rotors for new relations."""
        model = self.store.model
        m = model.cfg.m
        for _ in range(n_new_rels):
            rotor = CayleyRotor(m).to(device)
            model.relation_rotors.append(rotor)
        model.num_relations = len(model.relation_rotors)

    # =================================================================
    # Incremental training (runs in background thread)
    # =================================================================

    def _incremental_train(self, triple_indices: list[tuple],
                            n_existing_ent: int, n_existing_rel: int,
                            n_new_ent: int, n_new_rel: int,
                            device: torch.device) -> int:
        """Riemannian GD on new entities + new rotors, freeze existing.

        Trains new Cayley rotors and new entity embeddings to minimize
        geodesic distance on the new triples.
        """
        model = self.store.model
        P_v = model.dual_emb.P_v.detach()

        # Freeze all existing params
        for p in model.parameters():
            p.requires_grad_(False)

        # Collect learnable params: new rotors + new entity embeddings
        learnable_params = []

        # New rotor parameters
        for i in range(n_existing_rel, n_existing_rel + n_new_rel):
            for p in model.relation_rotors[i].parameters():
                p.requires_grad_(True)
                learnable_params.append(p)

        # New entity embedding parameters
        new_v_params = None
        if n_new_ent > 0:
            new_v_params = nn.Parameter(
                model.dual_emb.v_embed.weight.data[n_existing_ent:].clone()
            )
            learnable_params.append(new_v_params)

        if not learnable_params:
            for p in model.parameters():
                p.requires_grad_(True)
            return 0

        optimizer = torch.optim.Adam(learnable_params, lr=self.config.lr_write)

        triple_h = torch.tensor([t[0] for t in triple_indices], device=device)
        triple_r = torch.tensor([t[1] for t in triple_indices], device=device)
        triple_t = torch.tensor([t[2] for t in triple_indices], device=device)

        def _get_v(idx_t):
            """Get v-projection, differentiable for new entities."""
            vecs = []
            for idx in idx_t.tolist():
                if new_v_params is not None and idx >= n_existing_ent:
                    v = new_v_params[idx - n_existing_ent]
                else:
                    v = model.dual_emb.v_embed.weight[idx].detach()
                vecs.append(normalize(v @ P_v.T))
            return torch.stack(vecs)

        n_steps = self.config.n_steps
        for step in range(n_steps):
            optimizer.zero_grad()

            v_h = _get_v(triple_h)
            v_t = _get_v(triple_t)

            # Build rotor matrices — new rotors are differentiable
            R_list = []
            for r_idx in triple_r.tolist():
                R_list.append(model.relation_rotors[r_idx].get_rotor())
            R_batch = torch.stack(R_list)

            v_hr = normalize(torch.bmm(R_batch, v_h.unsqueeze(-1)).squeeze(-1))
            loss = geodesic_distance(v_hr, v_t).mean()

            loss.backward()
            optimizer.step()

        # Write optimized entity embeddings back
        if new_v_params is not None:
            with torch.no_grad():
                model.dual_emb.v_embed.weight.data[n_existing_ent:] = new_v_params.data

        # Unfreeze all
        for p in model.parameters():
            p.requires_grad_(True)

        return n_steps

    # =================================================================
    # Helpers
    # =================================================================

    def _resolve_triples(self, new_triples: list[ExtractedTriple],
                          adapter) -> list[tuple]:
        """Convert ExtractedTriple list to (h_idx, r_idx, t_idx) tuples."""
        triple_indices = []
        for t in new_triples:
            h_idx = adapter.entity_to_idx.get(t.head)
            t_idx = adapter.entity_to_idx.get(t.tail)
            rel = t.relation.strip().lower().replace(" ", "_")
            r_idx = adapter.relation_to_idx.get(rel)
            if h_idx is not None and t_idx is not None and r_idx is not None:
                triple_indices.append((h_idx, r_idx, t_idx))
        return triple_indices
