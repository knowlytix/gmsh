# SPDX-License-Identifier: Apache-2.0
"""Unified tool registry: register GMS + MoDeva + custom tools in one place.

Each tool is a ToolDef with:
  - name: tool identifier
  - description: for the LLM
  - input_schema: JSON Schema for parameters
  - handler: callable(args, context) -> (str, Any|None)
  - category: for grouping (gms, modeva, memory, custom)

The registry produces Anthropic tool_use format and dispatches calls.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from typing import Any, Callable

from agents.ds_agent.tools import (
    TOOL_DEFINITIONS as MODEVA_TOOL_DEFS,
    SessionState,
    execute_tool as modeva_execute,
)


@dataclass
class ToolDef:
    """A single tool definition."""
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict, Any], tuple[str, Any | None]]
    category: str = "custom"


class ToolRegistry:
    """Pluggable tool registry that unifies GMS, MoDeva, and custom tools."""

    def __init__(self):
        self._tools: dict[str, ToolDef] = {}

    def register(self, tool: ToolDef):
        self._tools[tool.name] = tool

    def register_fn(self, name: str, description: str, input_schema: dict,
                    handler: Callable, category: str = "custom"):
        self.register(ToolDef(
            name=name, description=description,
            input_schema=input_schema, handler=handler,
            category=category,
        ))

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def list_by_category(self, category: str) -> list[ToolDef]:
        return [t for t in self._tools.values() if t.category == category]

    def to_anthropic_tools(self) -> list[dict]:
        """Convert to Anthropic tool_use API format."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
        ]

    def execute(self, name: str, arguments: dict, context: Any = None) -> tuple[str, Any | None]:
        tool = self._tools.get(name)
        if tool is None:
            return f"Unknown tool: {name}", None
        try:
            return tool.handler(arguments, context)
        except Exception as e:
            tb = traceback.format_exc()
            return f"ERROR in {name}: {e}\n{tb}", None

    # ----- Bulk registration helpers -----

    def register_modeva_tools(self, state: SessionState):
        """Register all MoDeva tools from ds_agent.tools.

        Each tool delegates to modeva_execute, passing the shared SessionState.
        """
        for tdef in MODEVA_TOOL_DEFS:
            if tdef["name"] == "recall_memory":
                continue  # handled separately by the harness

            def _make_handler(tool_name: str):
                def handler(args: dict, ctx: Any) -> tuple[str, Any | None]:
                    return modeva_execute(tool_name, args, state)
                return handler

            self.register(ToolDef(
                name=tdef["name"],
                description=tdef["description"],
                input_schema=tdef["input_schema"],
                handler=_make_handler(tdef["name"]),
                category="modeva",
            ))

    def register_gms_tools(self, stores_dir: str, device, model_name: str = "claude-sonnet-4-20250514"):
        """Register GMS expert system tools (create, ingest, query, search, lookup, status, list, use).

        These wrap the same logic as docgms.mcp_server but run in-process
        instead of over MCP stdio transport.
        """
        import os
        import time
        from knowlytix.doc.config import DocGMSConfig
        from knowlytix.doc.store import GMSExpertStore
        from knowlytix.doc.llm_backend import create_backend
        from knowlytix.doc.query import QueryEngine

        os.makedirs(stores_dir, exist_ok=True)

        # Shared mutable state for GMS tools (mirrors mcp_server._active)
        gms_state = {
            "name": None,
            "store": None,
            "engine": None,
            "config": None,
            "llm": None,
            "stores_dir": os.path.abspath(stores_dir),
            "device": device,
            "model_name": model_name,
        }

        def _load_store(name: str) -> bool:
            store_path = os.path.join(gms_state["stores_dir"], name)
            if not os.path.isdir(store_path):
                return False
            config = DocGMSConfig(store_path=store_path)
            config.convert.llm_model = gms_state["model_name"]
            store = GMSExpertStore(config, gms_state["device"])
            if not store.load():
                return False
            llm = create_backend(config.convert)
            engine = QueryEngine(store, llm, config)
            gms_state.update(name=name, store=store, engine=engine, config=config, llm=llm)
            return True

        def _create_store(name: str) -> GMSExpertStore:
            store_path = os.path.join(gms_state["stores_dir"], name)
            os.makedirs(store_path, exist_ok=True)
            config = DocGMSConfig(store_path=store_path)
            config.convert.llm_model = gms_state["model_name"]
            store = GMSExpertStore(config, gms_state["device"])
            llm = create_backend(config.convert)
            engine = QueryEngine(store, llm, config)
            gms_state.update(name=name, store=store, engine=engine, config=config, llm=llm)
            return store

        def _list_stores() -> list[dict]:
            stores = []
            sd = gms_state["stores_dir"]
            if not os.path.isdir(sd):
                return stores
            for name in sorted(os.listdir(sd)):
                sp = os.path.join(sd, name)
                if not os.path.isdir(sp):
                    continue
                info = {"name": name, "path": sp}
                meta_path = os.path.join(sp, "metadata.json")
                if os.path.isfile(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    info["documents"] = len(meta.get("documents", []))
                triples_path = os.path.join(sp, "triples.json")
                if os.path.isfile(triples_path):
                    with open(triples_path) as f:
                        info["triples"] = len(json.load(f))
                else:
                    info["triples"] = 0
                stores.append(info)
            return stores

        def _ensure_active() -> str | None:
            if gms_state["store"] is not None:
                return None
            stores = _list_stores()
            if len(stores) == 1:
                if _load_store(stores[0]["name"]):
                    return None
            if len(stores) == 0:
                return "No expert systems found. Use gms_create first."
            return f"Multiple stores: {[s['name'] for s in stores]}. Use gms_use to select."

        # Auto-load
        auto = _list_stores()
        if len(auto) == 1:
            _load_store(auto[0]["name"])
        elif len(auto) > 1:
            auto.sort(key=lambda s: os.path.getmtime(s["path"]), reverse=True)
            _load_store(auto[0]["name"])

        # --- Tool handlers ---

        def h_gms_create(args, ctx):
            name = args["name"].strip().replace(" ", "-").lower()
            desc = args.get("description", "")
            sp = os.path.join(gms_state["stores_dir"], name)
            if os.path.isdir(sp):
                if _load_store(name):
                    return json.dumps({"status": "already_exists", "name": name}), None
            _create_store(name)
            meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "description": desc, "documents": []}
            os.makedirs(sp, exist_ok=True)
            with open(os.path.join(sp, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)
            return json.dumps({"status": "created", "name": name}), None

        def h_gms_list(args, ctx):
            stores = _list_stores()
            return json.dumps({"active": gms_state["name"], "stores": stores}), None

        def h_gms_use(args, ctx):
            name = args["name"].strip().lower()
            if _load_store(name):
                return json.dumps({"status": "switched", "name": name}), None
            return json.dumps({"error": f"'{name}' not found"}), None

        def h_gms_ingest(args, ctx):
            from knowlytix.doc.ingest import ingest_document
            doc_path = os.path.abspath(args["document_path"])
            if not os.path.isfile(doc_path):
                return json.dumps({"error": f"File not found: {doc_path}"}), None
            name = args.get("name", "") or gms_state["name"]
            if not name:
                name = os.path.splitext(os.path.basename(doc_path))[0].replace(" ", "-").lower()[:40]
            sp = os.path.join(gms_state["stores_dir"], name)
            if os.path.isdir(sp) and gms_state["name"] != name:
                _load_store(name)
            elif not os.path.isdir(sp):
                _create_store(name)
            config = gms_state["config"]
            config.train.epochs = args.get("epochs", 200)
            ingest_document(gms_state["store"], doc_path, gms_state["llm"], config, gms_state["device"])
            gms_state["engine"] = QueryEngine(gms_state["store"], gms_state["llm"], config)
            stats = gms_state["store"].stats()
            return json.dumps({"status": "success", "name": name, "store": stats}), None

        def h_gms_query(args, ctx):
            err = _ensure_active()
            if err:
                return json.dumps({"error": err}), None
            question = args["question"]
            mode = args.get("mode", "gms_llm")
            result = gms_state["engine"].query(question, mode=mode)
            return json.dumps({
                "answer": result.answer,
                "source": result.source,
                "confidence": round(result.confidence, 4),
            }), result

        def h_gms_search(args, ctx):
            err = _ensure_active()
            if err:
                return json.dumps({"error": err}), None
            engine = gms_state["engine"]
            if engine._geo_search is None:
                return json.dumps({"error": "No geometric search available"}), None
            concepts = [c.strip().lower() for c in args["concepts"].split(",") if c.strip()]
            if len(concepts) < 2:
                return json.dumps({"error": "Need at least 2 concepts"}), None
            groups = engine._group_entities_by_concept(concepts)
            if not groups:
                return json.dumps({"error": "No entities matched"}), None
            max_hops = args.get("max_hops", 5)
            result = engine._geo_search.connect_concept_groups(groups, max_hops)
            paths = [{"entities": p.entity_names, "relations": p.relation_names,
                       "geodesic": round(p.geodesic, 4)} for p in result.paths[:10]]
            return json.dumps({"paths_found": len(result.paths), "paths": paths}), None

        def h_gms_lookup(args, ctx):
            err = _ensure_active()
            if err:
                return json.dumps({"error": err}), None
            store = gms_state["store"]
            entity = args["entity"]
            matched = store.fuzzy_match_entity(entity)
            if not matched:
                return json.dumps({"error": f"No entity matching '{entity}'"}), None
            head_triples = [(r, t) for h, r, t in store.query_triples(head=matched) if r != "in_section"]
            tail_triples = [(h, r) for h, r, t in store.query_triples(tail=matched) if r != "in_section"]
            return json.dumps({
                "entity": matched,
                "as_head": [{"relation": r, "tail": t} for r, t in head_triples[:20]],
                "as_tail": [{"head": h, "relation": r} for h, r in tail_triples[:20]],
            }), None

        def h_gms_status(args, ctx):
            err = _ensure_active()
            if err:
                return json.dumps({"error": err}), None
            stats = gms_state["store"].stats()
            return json.dumps({"name": gms_state["name"], **stats}), None

        # --- Register all ---

        gms_tools = [
            ("gms_create", "Create a new GMS expert system.",
             {"type": "object", "properties": {
                 "name": {"type": "string", "description": "Expert system name"},
                 "description": {"type": "string", "description": "What it covers"},
             }, "required": ["name"]}, h_gms_create),

            ("gms_list", "List all available GMS expert systems.",
             {"type": "object", "properties": {}}, h_gms_list),

            ("gms_use", "Switch active expert system.",
             {"type": "object", "properties": {
                 "name": {"type": "string"},
             }, "required": ["name"]}, h_gms_use),

            ("gms_ingest", "Ingest a document (PDF/TeX/Markdown) into the active expert system.",
             {"type": "object", "properties": {
                 "document_path": {"type": "string", "description": "Path to document"},
                 "name": {"type": "string", "description": "Expert system name (optional)"},
                 "epochs": {"type": "integer", "description": "Training epochs (default 200)"},
             }, "required": ["document_path"]}, h_gms_ingest),

            ("gms_query", "Query the active expert system with a question.",
             {"type": "object", "properties": {
                 "question": {"type": "string"},
                 "mode": {"type": "string", "enum": ["gms_only", "llm_only", "gms_llm"]},
             }, "required": ["question"]}, h_gms_query),

            ("gms_search", "Find geodesic paths connecting concepts in the knowledge store.",
             {"type": "object", "properties": {
                 "concepts": {"type": "string", "description": "Comma-separated concepts"},
                 "max_hops": {"type": "integer"},
             }, "required": ["concepts"]}, h_gms_search),

            ("gms_lookup", "Look up everything known about an entity.",
             {"type": "object", "properties": {
                 "entity": {"type": "string"},
             }, "required": ["entity"]}, h_gms_lookup),

            ("gms_status", "Show active expert system statistics.",
             {"type": "object", "properties": {}}, h_gms_status),
        ]

        for name, desc, schema, handler in gms_tools:
            self.register(ToolDef(
                name=name, description=desc, input_schema=schema,
                handler=handler, category="gms",
            ))

        # Expose gms_state for the harness to access
        self._gms_state = gms_state
