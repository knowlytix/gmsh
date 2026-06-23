# SPDX-License-Identifier: Apache-2.0
"""Agent memory wrapper around GMSExpertStore + RuntimeLearner + ENM.

Provides an agent-friendly API for storing and recalling knowledge
accumulated across experimental sessions.  All heavy lifting is
delegated to the existing GMS infrastructure:

  - GMSExpertStore: persistent geometric knowledge graph + ENM
  - RuntimeLearner: two-phase (sync graph + async Riemannian GD) learning
  - ExactNumericalMemory: lossless IEEE 754 key-value register with SHA-256

Built in Chapter 5 of "How to Build Policy-Governed AI Agents."
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

from knowlytix.knowledge.config import DocGMSConfig
from knowlytix.knowledge.extract import ExtractedTriple
from knowlytix.knowledge.ingest import ingest_document
from knowlytix.knowledge.learn import LearnResult, RuntimeLearner
from knowlytix.knowledge.store import GMSExpertStore


@dataclass
class CompareResult:
    """Result of comparing two entities in geometric memory."""
    entity_a: str
    entity_b: str
    tension_energy: float | None = None
    interpretation: str = ""
    metric_deltas: dict = field(default_factory=dict)


class AgentMemory:
    """High-level memory interface for governed agents.

    Wraps GMSExpertStore, RuntimeLearner and ENM behind a simple API
    that an agent (or notebook) can call without knowing the geometric
    internals.

    Typical lifecycle::

        mem = AgentMemory("my_store")          # create or load
        mem.store_findings(triples, enm)       # after each experiment
        mem.recall_entity("LogisticRegression") # structured recall
        mem.compare_entities("RF", "GBT")      # geometric comparison
        mem.save()                             # persist to disk

    Parameters
    ----------
    store_path : str
        Directory for the on-disk store.
    knowledge_path : str, optional
        Path to a markdown knowledge document to ingest on first creation.
    device : str
        PyTorch device string ("cuda" or "cpu").
    """

    def __init__(
        self,
        store_path: str,
        knowledge_path: str | None = None,
        device: str = "cuda",
        epochs: int = 500,
    ):
        self.device = torch.device(
            device if torch.cuda.is_available() or device == "cpu" else "cpu"
        )

        # Build config.  AgentMemory always ingests with llm=None, so
        # default ingest_mode to "regex" -- the library default
        # ("hybrid") would raise because no LLMBackend is wired in.
        self._config = DocGMSConfig(store_path=store_path)
        self._config.train.epochs = epochs
        self._config.ingest_mode = "regex"

        # Core GMS components
        self.store = GMSExpertStore(self._config, device=self.device)
        self.learner: RuntimeLearner | None = None

        # Try to load existing store, else ingest knowledge document
        if self.store.exists():
            self.store.load()
            self.learner = RuntimeLearner(
                self.store, self._config.learn,
            )
        elif knowledge_path is not None:
            ingest_document(
                self.store, knowledge_path,
                llm=None, config=self._config, device=self.device,
            )
            self.learner = RuntimeLearner(
                self.store, self._config.learn,
            )
        else:
            # Empty store -- will be populated by store_findings()
            self._bootstrap_empty_store()
            self.learner = RuntimeLearner(
                self.store, self._config.learn,
            )

    # ----------------------------------------------------------------
    # Bootstrap
    # ----------------------------------------------------------------

    @classmethod
    def bootstrap(
        cls,
        store_path: str,
        seed_triples: list[tuple[str, str, str]],
        enm_entries: list[tuple[str, str, float]] | None = None,
        declarations: list[tuple[str, str, str]] | None = None,
        knowledge_path: str | None = None,
        device: str = "cuda",
        epochs: int = 500,
    ) -> "AgentMemory":
        """One-shot training of a findings store with full loss stack.

        The ordinary ``store_findings`` path routes new triples through
        the ``RuntimeLearner`` incremental update — 50 Riemannian-GD
        steps on only the new embeddings, with hard-negatives and
        tension pairs disabled. That is fine for small additions to an
        already-trained store, but it cannot teach the store to
        discriminate agreement from contradiction when the entities in
        question (e.g.\\ a batch of model-comparison findings) are
        entirely new.

        ``bootstrap`` instead builds one ``DocumentGraph`` containing
        the knowledge-doc triples (if any), all seed triples and all
        opposition declarations, then runs ``train_gms`` for
        ``epochs`` epochs with the full loss stack — margin loss,
        hard-negative mining and tension-pair mining driven by the
        ``opposite_of`` declarations.  The resulting embeddings
        discriminate sharply, so downstream ``GMSJudge.calibrate()``
        produces usable cuts.

        Parameters
        ----------
        store_path
            On-disk path for the persisted store.  Wiped if it exists.
        seed_triples
            Identity and findings triples to embed alongside the
            knowledge doc.
        enm_entries
            ``(category, entity_id, value)`` tuples written into the
            exact-numeric-memory register.
        declarations
            ``opposite_of`` triples (e.g.\\
            ``("passes", "opposite_of", "fails")``).  Read by
            ``mine_tension_pairs`` so the training loss can push
            declared contradictions toward :math:`E \\approx 2` and
            agreements toward :math:`E \\approx 0`.
        knowledge_path
            Optional markdown knowledge doc parsed before seeding.
        epochs
            Training epochs; 500 is the monograph default.

        Returns
        -------
        AgentMemory
            Fully trained, saved and ready for query / calibration.
        """
        from knowlytix.benchmark.graph import DocumentGraph
        from knowlytix.benchmark.ingest import ingest_markdown
        from knowlytix.core.graph.transport import RelationalTransport
        from knowlytix.core.memory.compression import CompressionMemory
        from knowlytix.core.memory.router import MemoryRouter
        from knowlytix.core.train_finstructbench import (
            GraphToGMS, populate_enm, train_gms,
        )

        dev = torch.device(
            device if torch.cuda.is_available() or device == "cpu" else "cpu"
        )

        if knowledge_path is not None:
            doc_graph = ingest_markdown(knowledge_path)
        else:
            doc_graph = DocumentGraph()

        for h, r, t in seed_triples:
            doc_graph.add_triple(h, r, t)
        for h, r, t in (declarations or []):
            doc_graph.add_triple(h, r, t)
        if enm_entries is not None:
            for category, entity_id, value in enm_entries:
                doc_graph.store_value(category, entity_id, float(value))

        import shutil
        if os.path.exists(store_path):
            shutil.rmtree(store_path)

        config = DocGMSConfig(store_path=store_path)
        config.train.epochs = epochs
        store = GMSExpertStore(config, device=dev)

        adapter = GraphToGMS(doc_graph)
        model = train_gms(
            adapter, dev,
            epochs=epochs,
            batch_size=config.train.batch_size,
            num_neg=config.train.neg_samples,
            lr=config.train.lr,
            lr_riemannian=config.train.lr_riemannian,
            geometry=config.geometry,
            loss=config.loss,
        )
        model.eval()

        enm = populate_enm(doc_graph, adapter)
        transport = RelationalTransport(model)
        compression = CompressionMemory(
            k=config.memory.k, d=config.geometry.m,
        )
        router = MemoryRouter(enm, compression)

        store.model = model
        store.adapter = adapter
        store.enm = enm
        store.doc_graph = doc_graph
        store.transport = transport
        store.compression = compression
        store.router = router
        store.save()

        mem = cls.__new__(cls)
        mem.device = dev
        mem._config = config
        mem.store = store
        mem.learner = RuntimeLearner(store, config.learn)
        return mem

    def _bootstrap_empty_store(self):
        """Create minimal GMS structures so learn_triples() works."""
        from knowlytix.benchmark.graph import DocumentGraph
        from knowlytix.core.graph.transport import RelationalTransport
        from knowlytix.core.memory.compression import CompressionMemory
        from knowlytix.core.memory.enm import ExactNumericalMemory
        from knowlytix.core.memory.router import MemoryRouter
        from knowlytix.core.train_finstructbench import GraphToGMS

        cfg = self._config.geometry

        # Create an empty document graph with a seed triple
        doc_graph = DocumentGraph()
        doc_graph.add_triple("__seed__", "initializes", "memory")

        adapter = GraphToGMS(doc_graph)

        # train_gms creates model internally and returns it
        from knowlytix.core.train_finstructbench import train_gms
        model = train_gms(adapter, self.device, epochs=5)

        self.store.model = model
        self.store.adapter = adapter
        self.store.doc_graph = doc_graph
        self.store.enm = ExactNumericalMemory(capacity=10000)
        self.store.transport = RelationalTransport(model)

        m = cfg.m
        self.store.compression = CompressionMemory(
            k=self._config.memory.k, d=m,
        )
        self.store.router = MemoryRouter(
            self.store.enm, self.store.compression,
        )

    # ----------------------------------------------------------------
    # Store findings
    # ----------------------------------------------------------------

    def store_findings(
        self,
        triples: list[tuple[str, str, str]],
        enm_entries: list[tuple[str, str, float]] | None = None,
    ) -> LearnResult:
        """Store experimental findings in geometric memory.

        Parameters
        ----------
        triples : list of (head, relation, tail)
            Knowledge triples to embed on the manifold.
        enm_entries : list of (category, entity_id, value), optional
            Exact numeric values to store with SHA-256 integrity.

        Returns
        -------
        LearnResult
            Outcome of the two-phase learning protocol.
        """
        # Convert to ExtractedTriple objects
        extracted = []
        for h, r, t in triples:
            extracted.append(ExtractedTriple(head=h, relation=r, tail=t))

        # Learn triples via RuntimeLearner (Phase 1: sync graph update)
        result = self.learner.learn_triples(extracted, self.device)

        # Flush training (Phase 2: background Riemannian GD)
        if result.training_queued:
            self.learner.flush_training()
            self.learner.wait_for_training(timeout=120.0)

        # Write ENM entries
        if enm_entries is not None:
            for category, entity_id, value in enm_entries:
                self.learner.learn_enm(category, entity_id, value)

        return result

    # ----------------------------------------------------------------
    # Recall
    # ----------------------------------------------------------------

    def recall_entity(self, entity: str) -> dict:
        """Retrieve everything known about an entity.

        Returns a dict with keys:
          - triples_as_head: triples where entity is the head
          - triples_as_tail: triples where entity is the tail
          - enm_values: dict of {category: value} from ENM
          - exists: whether entity has a geometric embedding
        """
        matched = self.store.fuzzy_match_entity(entity)
        if matched is None:
            return {
                "entity": entity,
                "matched": None,
                "exists": False,
                "triples_as_head": [],
                "triples_as_tail": [],
                "enm_values": {},
            }

        # Graph triples
        head_triples = self.store.query_triples(head=matched)
        tail_triples = self.store.query_triples(tail=matched)

        # ENM values
        enm_vals = {}
        if self.store.enm is not None:
            for key in self.store.enm.keys():
                if matched in key.id or entity in key.id:
                    val = self.store.lookup_enm(key.type, key.id)
                    if val is not None:
                        enm_vals[f"{key.type}::{key.id}"] = val

        return {
            "entity": entity,
            "matched": matched,
            "exists": self.store.entity_exists(matched),
            "triples_as_head": [(h, r, t) for h, r, t in head_triples],
            "triples_as_tail": [(h, r, t) for h, r, t in tail_triples],
            "enm_values": enm_vals,
        }

    def recall_metric(self, category: str, entity_id: str) -> float | None:
        """Recall an exact numeric value from ENM.

        Parameters
        ----------
        category : str
            The metric category (e.g. "test_accuracy", "test_auc").
        entity_id : str
            The entity identifier (e.g. "RandomForest/iris").

        Returns
        -------
        float or None
            The exact stored value, or None if not found.
        """
        return self.store.lookup_enm(category, entity_id)

    # ----------------------------------------------------------------
    # Compare
    # ----------------------------------------------------------------

    def compare_entities(self, entity_a: str, entity_b: str) -> CompareResult:
        """Compare two entities using geometric tension energy and ENM.

        Tension energy interpretation:
          - ~0: agreement (aligned embeddings)
          - ~1.414: irrelevance (orthogonal)
          - ~2: contradiction (antipodal)

        Returns
        -------
        CompareResult
            Contains tension_energy, interpretation and metric_deltas.
        """
        # Fuzzy match
        a_match = self.store.fuzzy_match_entity(entity_a) or entity_a
        b_match = self.store.fuzzy_match_entity(entity_b) or entity_b

        te = self.store.tension_energy(a_match, b_match)

        # Interpret
        if te is None:
            interp = "unknown (entity not found)"
        elif te < 0.8:
            interp = "agreement"
        elif te < 1.6:
            interp = "irrelevance"
        else:
            interp = "contradiction"

        # Compare ENM metrics
        deltas = {}
        if self.store.enm is not None:
            categories_a = {}
            categories_b = {}
            for key in self.store.enm.keys():
                if a_match in key.id or entity_a in key.id:
                    val = self.store.lookup_enm(key.type, key.id)
                    if val is not None:
                        categories_a[key.type] = val
                if b_match in key.id or entity_b in key.id:
                    val = self.store.lookup_enm(key.type, key.id)
                    if val is not None:
                        categories_b[key.type] = val
            # Compute deltas for shared categories
            for cat in set(categories_a) & set(categories_b):
                deltas[cat] = categories_a[cat] - categories_b[cat]

        return CompareResult(
            entity_a=a_match,
            entity_b=b_match,
            tension_energy=te,
            interpretation=interp,
            metric_deltas=deltas,
        )

    # ----------------------------------------------------------------
    # Exploration
    # ----------------------------------------------------------------

    def find_related(
        self, entity: str, relation: str, top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Find entities related to *entity* via *relation*.

        Uses link_predict (beam search on the manifold).

        Returns list of (entity_name, geodesic_distance).
        """
        matched = self.store.fuzzy_match_entity(entity)
        if matched is None:
            return []
        return self.store.link_predict(matched, relation, top_k=top_k)

    # ----------------------------------------------------------------
    # Contradiction detection
    # ----------------------------------------------------------------

    def check_contradiction(
        self, head: str, relation: str, tail: str,
    ) -> float:
        """Check how much a proposed triple contradicts existing knowledge.

        Returns tension energy between the proposed tail and the
        existing tail(s) for (head, relation, ?).  Higher means more
        contradictory.  Returns 0.0 if no comparison is possible.
        """
        matched_head = self.store.fuzzy_match_entity(head)
        if matched_head is None:
            return 0.0

        # Find existing tails for this (head, relation)
        existing = self.store.query_triples(head=matched_head, relation=relation)
        if not existing:
            return 0.0

        # Compute max tension between proposed tail and existing tails
        max_tension = 0.0
        matched_tail = self.store.fuzzy_match_entity(tail)
        if matched_tail is None:
            return 0.0

        for _, _, existing_tail in existing:
            te = self.store.tension_energy(matched_tail, existing_tail)
            if te is not None and te > max_tension:
                max_tension = te

        return max_tension

    # ----------------------------------------------------------------
    # Stats
    # ----------------------------------------------------------------

    def memory_stats(self) -> dict:
        """Return summary statistics of the memory store.

        Returns dict with: entities, relations, triples, enm_entries,
        store_path, training_status.
        """
        stats = self.store.stats()
        stats["training_status"] = (
            self.learner.training_status() if self.learner else "N/A"
        )
        return stats

    # ----------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------

    def save(self):
        """Save all memory state to disk."""
        self.store.save()

    def __repr__(self) -> str:
        s = self.store.stats()
        return (
            f"AgentMemory(entities={s['entities']}, "
            f"relations={s['relations']}, "
            f"triples={s['triples']}, "
            f"enm={s['enm_entries']}, "
            f"path='{s['store_path']}')"
        )
