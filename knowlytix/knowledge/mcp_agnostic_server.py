#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic MCP server over a GMS store (knowlytix-only tools).

Exposes the deterministic operations of :mod:`knowlytix.knowledge.mcp_tools` —
the memory primitives, geometric grounded retrieval, optional compiler-parsed
retrieval, and a geometric admissibility gate — over MCP stdio. The geometric
tools call no language model, so the server runs without an API key; the client
model reads the returned facts and writes the answer.

Source-only entry point (not shipped in the knowlytix-knowledge wheel, like
``mcp_server.py``), because it pulls the optional ``mcp`` dependency. The reusable
tool logic lives in the packaged :mod:`knowlytix.knowledge.mcp_tools`. For the
full server that also exposes forgeloop's governance/testing/ingest tools, see
``forgeloop.mcp``.

Usage:
    python -m knowlytix.knowledge.mcp_agnostic_server [--stores-dir DIR] [--device cpu]
"""

from __future__ import annotations

import argparse

from mcp.server.fastmcp import FastMCP

from knowlytix.knowledge.mcp_tools import ActiveStore, register_store_tools

_INSTRUCTIONS = (
    "Deterministic tools over a geometric-memory store; the geometric paths call "
    "no language model. Start with gms_status/gms_list. For a question, call "
    "rag_retrieve and synthesize only from its facts, abstaining when it abstains; "
    "if the store ships a query compiler, rag_retrieve_compiled binds more "
    "phrasings (model-backed). Look up specifics with gms_lookup / gms_query_triples "
    "/ gms_lookup_enm; clear an action with gate_plausibility."
)


def build_server(stores_dir: str, device=None) -> FastMCP:
    active = ActiveStore(stores_dir, device)
    mcp = FastMCP("GMS (geometric memory) — model-agnostic", instructions=_INSTRUCTIONS)
    register_store_tools(mcp, active)
    return mcp


def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--stores-dir", default="gms_stores")
    p.add_argument("--device", default=None)
    args, _ = p.parse_known_args()
    build_server(args.stores_dir, args.device).run(transport="stdio")


if __name__ == "__main__":
    main()
