# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic, deterministic tools over a GMS store.

Pure functions and a small :class:`StoreTools` facade returning JSON-able dicts
for the store's geometric operations: the memory primitives, geometric grounded
retrieval (retrieve-only, no answer generation), an optional compiler-parsed
retrieval, and a geometric admissibility gate. The geometric paths call **no
language model** — a client model reads the returned facts and writes the
answer. Only :meth:`StoreTools.rag_retrieve_compiled` loads a model (the store's
trained query compiler), and only when the caller opts into it.

This module has no MCP / FastMCP dependency, so it ships in the wheel and is
importable without the server stack. A server (``mcp_agnostic_server`` here, or
``forgeloop.mcp``) is a thin wrapper that exposes these as tools.

The loader :func:`load_store_auto` reads the persisted checkpoint's embedding
dimensions before constructing the model, because :meth:`GMSExpertStore.load`
builds from ``config.geometry`` and does not read the geometry back from disk,
so a default config silently mismatches any store not built at the default size.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import sys

import torch

from knowlytix.core.config import GeometryConfig
from knowlytix.knowledge.config import DocGMSConfig
from knowlytix.knowledge.llm_backend import LLMBackend
from knowlytix.knowledge.store import GMSExpertStore

__all__ = ["NoLLM", "infer_geometry", "load_store_auto", "StoreTools",
           "ActiveStore", "register_store_tools"]


class NoLLM(LLMBackend):
    """A backend that refuses to generate.

    ``RagConfig`` requires an ``llm`` field, but the retrieve-only geometric path
    never calls it. This stub satisfies the type and fails loudly if any path
    tries to generate, so the model-free contract cannot be broken silently.
    """

    def call(self, system: str, user: str, max_tokens: int = 2048) -> str:
        raise RuntimeError(
            "model-agnostic tools do not generate; the client model synthesizes "
            "from the returned facts")

    @property
    def model_name(self) -> str:
        return "none"


def infer_geometry(store_dir: str) -> GeometryConfig:
    """Read the embedding dimensions off a persisted store's ``model.pt``.

    The v/u-embedding widths give ``d_v``/``d_u``, the projector row count gives
    ``m``, and a relation rotor block gives ``d``.
    """
    sd = torch.load(os.path.join(store_dir, "model.pt"),
                    map_location="cpu", weights_only=True)
    return GeometryConfig(
        d_v=int(sd["dual_emb.v_embed.weight"].shape[1]),
        d_u=int(sd["dual_emb.u_embed.weight"].shape[1]),
        m=int(sd["dual_emb.P_v"].shape[0]),
        d=int(sd["relation_rotors.0._W"].shape[0]),
    )


def load_store_auto(store_dir: str, device=None) -> GMSExpertStore:
    """Load a persisted store, inferring its geometry from the checkpoint.

    Works regardless of the dimensions the store was built at. Returns the loaded
    :class:`GMSExpertStore`; raises if the directory is not a valid store.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DocGMSConfig(store_path=store_dir)
    try:
        config.geometry = infer_geometry(store_dir)
    except Exception:  # noqa: BLE001 — fall back to config default dims
        pass
    store = GMSExpertStore(config, dev)
    if not store.load():
        raise FileNotFoundError(f"no GMS store at {store_dir}")
    return store


class StoreTools:
    """Deterministic tool operations bound to one loaded store.

    Args:
        store: a loaded :class:`GMSExpertStore`.
        store_dir: the store directory (needed to wire tuned encoders / the
            calibrated relevance gate / the query compiler).
        device: torch device for the query compiler, if used.
    """

    def __init__(self, store: GMSExpertStore, store_dir: str, device=None):
        self.store = store
        self.store_dir = store_dir
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self._rag = None
        self._rag_compiled = None

    # ---- retrieval pipelines (lazy) --------------------------------------

    def _geometric_pipe(self):
        if self._rag is not None:
            return self._rag
        from knowlytix.knowledge.rag import RagConfig, RagPipeline
        overrides = dict(query_parse_mode="geometric", verify_mode="geometric",
                         verify_llm_output=False)
        if os.path.isdir(os.path.join(self.store_dir, "contradiction_encoder")):
            overrides.update(relevance_gate=True, relevance_mode="geometric")
        else:
            overrides.update(relevance_gate=False)
        cfg = RagConfig.from_store_dir(self.store_dir, llm=NoLLM(), **overrides)
        self._rag = RagPipeline.from_store(self.store, cfg)
        return self._rag

    def _compiled_pipe(self):
        if self._rag_compiled is not None:
            return self._rag_compiled
        if not os.path.isdir(os.path.join(self.store_dir, "query_compiler")):
            return None
        from knowlytix.knowledge.rag import RagConfig, RagPipeline
        cfg = RagConfig.from_store_dir(self.store_dir, llm=NoLLM(),
                                       compiler_device=str(self.device))
        self._rag_compiled = RagPipeline.from_store(self.store, cfg)
        return self._rag_compiled

    @staticmethod
    def _answer_dict(ans, top_k: int, parse_mode: str) -> dict:
        return {
            "parse_mode": parse_mode,
            "decision": ans.decision,
            "route": ans.route,
            "confidence": round(float(ans.confidence), 4),
            "query_triples": [t.as_tuple() for t in ans.query_triples],
            "bound_triples": [[b.head, b.relation, b.tail, b.bound]
                              for b in ans.bound_triples],
            "sources": [{"triple": [f.head, f.relation, f.tail], "source": f.source,
                         "score": f.score, "location": f.location}
                        for f in ans.sources][:top_k],
            "notice": ans.notice,
        }

    # ---- memory primitives (deterministic geometry) ----------------------

    def status(self) -> dict:
        stats = self.store.stats()
        ents, rels = [], []
        if getattr(self.store, "adapter", None):
            ents = sorted(self.store.adapter.entity_to_idx.keys(), key=len)[:20]
            rels = list(self.store.adapter.relation_to_idx.keys())[:20]
        return {**stats, "sample_entities": ents, "sample_relations": rels}

    def score_triple(self, head: str, relation: str, tail: str) -> dict:
        return {"triple": [head, relation, tail],
                "score": self.store.score_triple(head, relation, tail)}

    def query_triples(self, head: str = "", relation: str = "", tail: str = "") -> dict:
        res = self.store.query_triples(head=head or None, relation=relation or None,
                                       tail=tail or None)
        triples = [[h, r, t] for h, r, t in res if r != "in_section"]
        return {"count": len(triples), "triples": triples[:100]}

    def link_predict(self, head: str, relation: str, top_k: int = 10) -> dict:
        preds = self.store.link_predict(head, relation, top_k=top_k)
        return {"head": head, "relation": relation,
                "predictions": [[t, sc] for t, sc in preds]}

    def lookup_enm(self, category: str, entity_id: str) -> dict:
        return {"category": category, "entity_id": entity_id,
                "value": self.store.lookup_enm(category, entity_id)}

    def tension(self, entity_a: str, entity_b: str) -> dict:
        return {"a": entity_a, "b": entity_b,
                "tension": self.store.tension_energy(entity_a, entity_b)}

    def holonomy(self, relation_path: list[str], direct_relation: str) -> dict:
        return {"path": relation_path, "direct": direct_relation,
                "holonomy": self.store.check_holonomy(relation_path, direct_relation)}

    def lookup(self, entity: str) -> dict:
        matched = self.store.fuzzy_match_entity(entity) or entity
        as_head = [[r, t] for h, r, t in self.store.query_triples(head=matched)
                   if r != "in_section"]
        as_tail = [[h, r] for h, r, t in self.store.query_triples(tail=matched)
                   if r != "in_section"]
        return {"entity": matched, "as_head": as_head[:40], "as_tail": as_tail[:40],
                "total": len(as_head) + len(as_tail)}

    def contradictions(self) -> dict:
        c = self.store.find_contradictions()
        return {"count": len(c), "contradictions": c[:50]}

    # ---- retrieval + gate ------------------------------------------------

    def rag_retrieve(self, question: str, top_k: int = 5) -> dict:
        ans = self._geometric_pipe().query(question, generate=False)
        return {"question": question, **self._answer_dict(ans, top_k, "geometric")}

    def rag_retrieve_compiled(self, question: str, top_k: int = 5) -> dict:
        pipe = self._compiled_pipe()
        if pipe is None:
            return {"error": "store has no trained query compiler (query_compiler/); "
                             "use rag_retrieve (geometric)."}
        ans = pipe.query(question, generate=False)
        return {"question": question, **self._answer_dict(ans, top_k, "compiler")}

    def gate_plausibility(self, tool: str, context: str = "default",
                          relation: str = "should_call", theta: float = 1.5) -> dict:
        score = self.store.score_triple(context, relation, tool)
        if score is None:
            decision, reason = "allow", "no geometry for this edge; defaulting to allow"
        elif score <= theta:
            decision, reason = "allow", f"plausibility {score:.3f} <= theta {theta}"
        else:
            decision, reason = "deny", f"plausibility {score:.3f} > theta {theta}"
        return {"tool": tool, "context": context, "relation": relation,
                "score": score, "theta": theta, "decision": decision, "reason": reason}


# ---------------------------------------------------------------------------
# Multi-store selection + FastMCP registration (no MCP import here: the caller
# passes the FastMCP instance, so this stays a plain-library module that ships
# in the wheel without forcing the mcp dependency).
# ---------------------------------------------------------------------------

class ActiveStore:
    """Selects and holds one active store among the subdirectories of a dir."""

    def __init__(self, stores_dir: str, device=None):
        self.stores_dir = os.path.abspath(stores_dir)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.stores_dir, exist_ok=True)
        self.name: str | None = None
        self.tools: StoreTools | None = None

    def list_stores(self) -> list[dict]:
        out = []
        for name in sorted(os.listdir(self.stores_dir)):
            path = os.path.join(self.stores_dir, name)
            if not os.path.isdir(path):
                continue
            info = {"name": name, "path": path}
            tp = os.path.join(path, "triples.json")
            if os.path.isfile(tp):
                with open(tp) as f:
                    info["triples"] = len(json.load(f))
            out.append(info)
        return out

    def use(self, name: str) -> bool:
        path = os.path.join(self.stores_dir, name.strip())
        if not os.path.isdir(path):
            return False
        store = load_store_auto(path, self.device)
        self.name = name.strip()
        self.tools = StoreTools(store, path, self.device)
        return True

    def ensure(self) -> str | None:
        """Auto-load a single store; return an error message if selection is needed."""
        if self.tools is not None:
            return None
        stores = self.list_stores()
        if len(stores) == 1 and self.use(stores[0]["name"]):
            return None
        if not stores:
            return f"No stores under {self.stores_dir}."
        return (f"Multiple stores: {[s['name'] for s in stores]}. Call gms_use.")


def register_store_tools(mcp, active: ActiveStore) -> None:
    """Register the model-agnostic store tools on a FastMCP instance.

    ``mcp`` is a ``mcp.server.fastmcp.FastMCP`` (passed in so this module needs no
    MCP import). Each tool routes library stdout to stderr, because the MCP stdio
    transport uses stdout for JSON-RPC and knowlytix prints progress there.
    """
    def _tool(fn):
        @functools.wraps(fn)
        def wrapped(*a, **k):
            with contextlib.redirect_stdout(sys.stderr):
                return fn(*a, **k)
        return mcp.tool()(wrapped)

    def _dumps(d):
        return json.dumps(d, indent=2, default=str)

    def _need():
        err = active.ensure()
        return err

    @_tool
    def gms_list() -> str:
        """List available stores and which is active."""
        return _dumps({"active": active.name, "stores_dir": active.stores_dir,
                       "stores": active.list_stores()})

    @_tool
    def gms_use(name: str) -> str:
        """Switch the active store (see gms_list)."""
        if active.use(name):
            return _dumps({"status": "switched", "active": active.name,
                           **active.tools.status()})
        return _dumps({"error": f"store '{name}' not found",
                       "available": [s["name"] for s in active.list_stores()]})

    @_tool
    def gms_status() -> str:
        """Statistics and sample vocabulary for the active store."""
        err = _need()
        return _dumps({"error": err} if err else {"active": active.name,
                                                   **active.tools.status()})

    @_tool
    def gms_score_triple(head: str, relation: str, tail: str) -> str:
        """Geodesic plausibility of a (head, relation, tail); lower = more plausible."""
        err = _need()
        return _dumps({"error": err} if err else active.tools.score_triple(head, relation, tail))

    @_tool
    def gms_query_triples(head: str = "", relation: str = "", tail: str = "") -> str:
        """Structural pattern match over stored triples (empty fields = wildcards)."""
        err = _need()
        return _dumps({"error": err} if err else active.tools.query_triples(head, relation, tail))

    @_tool
    def gms_link_predict(head: str, relation: str, top_k: int = 10) -> str:
        """Rank the most plausible tails for (head, relation)."""
        err = _need()
        return _dumps({"error": err} if err else active.tools.link_predict(head, relation, top_k))

    @_tool
    def gms_lookup_enm(category: str, entity_id: str) -> str:
        """Exact numeric recall from the ENM register."""
        err = _need()
        return _dumps({"error": err} if err else active.tools.lookup_enm(category, entity_id))

    @_tool
    def gms_tension(entity_a: str, entity_b: str) -> str:
        """Contradiction signal between two entities in [0, 2]."""
        err = _need()
        return _dumps({"error": err} if err else active.tools.tension(entity_a, entity_b))

    @_tool
    def gms_holonomy(relation_path: list[str], direct_relation: str) -> str:
        """Multi-hop path consistency vs a direct relation."""
        err = _need()
        return _dumps({"error": err} if err else active.tools.holonomy(relation_path, direct_relation))

    @_tool
    def gms_lookup(entity: str) -> str:
        """Everything the store knows about an entity (triples + ENM)."""
        err = _need()
        return _dumps({"error": err} if err else active.tools.lookup(entity))

    @_tool
    def gms_contradictions() -> str:
        """Internally contradictory entity pairs the store detects."""
        err = _need()
        return _dumps({"error": err} if err else active.tools.contradictions())

    @_tool
    def rag_retrieve(question: str, top_k: int = 5) -> str:
        """Grounded facts + accept/abstain for a question (geometric parse, no model).

        Synthesize the answer only from the returned facts; abstain when the tool
        abstains.
        """
        err = _need()
        return _dumps({"error": err} if err else active.tools.rag_retrieve(question, top_k))

    @_tool
    def rag_retrieve_compiled(question: str, top_k: int = 5) -> str:
        """Grounded retrieval via the store's trained NL->query compiler (MODEL-BACKED).

        Binds far more phrasings than the geometric parser, but loads the compiler
        model on first use, so unlike the rest of these tools it is not
        model-agnostic. Errors if the store has no query_compiler/.
        """
        err = _need()
        return _dumps({"error": err} if err else active.tools.rag_retrieve_compiled(question, top_k))

    @_tool
    def gate_plausibility(tool: str, context: str = "default",
                          relation: str = "should_call", theta: float = 1.5) -> str:
        """Geometric admissibility of calling a tool in a context; 'deny' = stop."""
        err = _need()
        return _dumps({"error": err} if err else
                      active.tools.gate_plausibility(tool, context, relation, theta))
