#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for the Harness.

Usage:
    # Interactive mode
    python -m harness.run

    # Single request
    python -m harness.run --request "Load BikeSharing, train XGB depth-2, run FANOVA"

    # With specific GMS store
    python -m harness.run --stores-dir gms_stores --session-store harness_session

    # MoDeva-only (no GMS tools)
    python -m harness.run --no-gms

    # GMS-only (no MoDeva tools)
    python -m harness.run --no-modeva
"""

import argparse
import os
import sys

# Ensure project root is on path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from knowlytix.harness.core import Harness, HarnessConfig


def main():
    parser = argparse.ArgumentParser(description="GMS + MoDeva Agent Harness")
    parser.add_argument("--request", "-r", type=str, default="",
                        help="Single request to process (non-interactive)")
    parser.add_argument("--model", type=str, default="claude-opus-4-7",
                        help="Claude model to use")
    parser.add_argument("--stores-dir", type=str, default="gms_stores",
                        help="Directory for GMS expert system stores")
    parser.add_argument("--session-store", type=str, default="harness_session_store",
                        help="Path for session GMS store")
    parser.add_argument("--modeva-store", type=str, default="",
                        help="Path to pre-trained MoDeva GMS store")
    parser.add_argument("--no-gms", action="store_true",
                        help="Skip registering GMS tools")
    parser.add_argument("--no-modeva", action="store_true",
                        help="Skip registering MoDeva tools")
    parser.add_argument("--no-learn", action="store_true",
                        help="Disable auto-learning into session GMS")
    parser.add_argument("--max-turns", type=int, default=30,
                        help="Maximum tool call turns per request")
    parser.add_argument("--device", type=str, default="",
                        help="Device for GMS embeddings (e.g., 'cpu' when GPU is used by vLLM)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max output tokens per LLM call")

    args = parser.parse_args()

    config = HarnessConfig(
        model=args.model,
        stores_dir=args.stores_dir,
        session_store_path=args.session_store,
        modeva_store_path=args.modeva_store,
        auto_learn=not args.no_learn,
        max_turns=args.max_turns,
        device=args.device,
        max_tokens=args.max_tokens,
    )

    print("=" * 60)
    print("  GMS + MoDeva Agent Harness")
    print("=" * 60)

    harness = Harness(config)

    if not args.no_gms:
        print("  Registering GMS tools...")
        harness.register_gms_tools()

    if not args.no_modeva:
        print("  Registering MoDeva tools...")
        harness.register_modeva_tools()

    print(f"  Tools registered: {len(harness.tools)}")
    print(f"  Categories: {', '.join(sorted(set(t.category for t in harness.registry._tools.values())))}")
    print("=" * 60)

    if args.request:
        # Single request mode
        response = harness.run(args.request)
        print(f"\n{'─' * 60}")
        print(response)
        harness.save()
        return

    # Interactive mode
    print("\nInteractive mode. Type 'quit' to exit, 'reset' to clear history.")
    print("Type 'tools' to list available tools, 'stats' for GMS stats.\n")

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            harness.reset()
            print("  [Harness] Conversation reset.")
            continue
        if user_input.lower() == "tools":
            for name in harness.tools:
                t = harness.registry.get(name)
                print(f"  [{t.category:8s}] {name}: {t.description[:60]}")
            continue
        if user_input.lower() == "stats":
            print(f"  Session: {harness.session_stats}")
            continue
        if user_input.lower() == "save":
            harness.save()
            continue

        response = harness.run(user_input)
        print(f"\nAgent> {response}\n")

    harness.save()
    print("  [Harness] Session saved. Goodbye.")


if __name__ == "__main__":
    main()
