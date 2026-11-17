# SPDX-License-Identifier: Apache-2.0
"""One-shot builder for the shared GMS fixture stores used by the tutorials."""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

# Resolve paths relative to this file.  Notebooks ship inside the
# knowlytix-harness wheel as of the 2026-05-07 relocation, so this file
# lives at:
#   knowlytix/harness/governance/tutorials/notebooks/build_fixture.py
TUTORIALS_ROOT = Path(__file__).resolve().parent.parent  # .../governance/tutorials/
REPO_ROOT = TUTORIALS_ROOT.parents[3]  # 4 up: tutorials → governance → harness → knowlytix → repo

# Per Q1=A (ship in repo, see governance_tutorials_relocation_plan.md):
# the prebuilt fixtures live alongside the notebooks at the in-package
# path so they ship transitively in the wheel and are available without
# a separate build step.  Override via KNOWLYTIX_GOVERNANCE_FIXTURES_DIR
# if you're rebuilding from source for development.
_env_dir = os.environ.get("KNOWLYTIX_GOVERNANCE_FIXTURES_DIR")
FIXTURES_DIR = Path(_env_dir).expanduser() if _env_dir else TUTORIALS_ROOT / "fixtures"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowlytix.harness.governance.tutorials.notebooks._shared import quiet  # noqa: E402

BUILD_SEED = 42
INGEST_EPOCHS = 500
GEOMETRY_M = 128

KNOWLEDGE = [
    {
        "name": "ds_practices_store",
        "knowledge_path": REPO_ROOT / "knowlytix" / "harness" / "governance"
        / "knowledge" / "data_science_practices.md",
    },
    {
        "name": "california_housing_store",
        "knowledge_path": REPO_ROOT / "knowlytix" / "harness" / "governance"
        / "knowledge" / "california_housing_domain.md",
    },
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_one(entry: dict, device: torch.device, rebuild: bool) -> dict:
    from knowlytix.core.config import TrainConfig
    from knowlytix.knowledge.config import DocGMSConfig
    from knowlytix.knowledge.ingest import ingest_document
    from knowlytix.knowledge.llm_backend import AnthropicBackend
    from knowlytix.knowledge.store import GMSExpertStore

    name = entry["name"]
    knowledge_path = entry["knowledge_path"]
    store_dir = FIXTURES_DIR / name

    if not knowledge_path.exists():
        raise FileNotFoundError(f"Knowledge markdown missing: {knowledge_path}")

    if store_dir.exists() and not rebuild:
        print(f"[skip] {name} already exists at {store_dir}")
        return {"skipped": True, "name": name, "store_path": str(store_dir),
                "source_markdown_path": str(knowledge_path),
                "source_markdown_sha256": _sha256(knowledge_path)}

    if rebuild and store_dir.exists():
        print(f"[rebuild] removing {store_dir}")
        shutil.rmtree(store_dir)

    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if has_key:
        ingest_mode = "hybrid"
        llm = AnthropicBackend()
        print(f"[build] {name} using mode=hybrid (ANTHROPIC_API_KEY present)")
    else:
        ingest_mode = "regex"
        llm = None
        print(f"[build] {name} using mode=regex (no ANTHROPIC_API_KEY)")

    torch.manual_seed(BUILD_SEED)

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    # Let DocGMSConfig apply its own geometry defaults (d_v=256, d_u=256,
    # m=128, d=128) so the fixture state_dict matches what AgentMemory
    # constructs when it loads via the same defaulting path. Passing an
    # explicit GeometryConfig(m=GEOMETRY_M) here would silently use the
    # bare GeometryConfig defaults (d_v=384) and produce a fixture that
    # AgentMemory cannot load (size mismatch on dual_emb.P_v).
    config = DocGMSConfig(
        store_path=str(store_dir),
        train=TrainConfig(epochs=INGEST_EPOCHS, device=str(device),
                          seed=BUILD_SEED),
        ingest_mode=ingest_mode,
    )

    t0 = time.time()
    store = GMSExpertStore(config, device=device)
    with quiet():
        ingest_document(store, str(knowledge_path), llm=llm,
                        config=config, device=device)
    elapsed = time.time() - t0

    stats = store.stats()
    summary = {
        "skipped": False,
        "name": name,
        "store_path": str(store_dir),
        "ingest_mode": ingest_mode,
        "ingest_epochs": INGEST_EPOCHS,
        "geometry_m": GEOMETRY_M,
        "n_triples": int(stats.get("triples", 0)),
        "n_enm": int(stats.get("enm_entries", 0)),
        "n_entities": int(stats.get("entities", 0)),
        "n_relations": int(stats.get("relations", 0)),
        "build_seconds": round(elapsed, 2),
        "source_markdown_path": str(knowledge_path),
        "source_markdown_sha256": _sha256(knowledge_path),
    }
    print(
        f"[done] {name}: triples={summary['n_triples']} "
        f"enm={summary['n_enm']} entities={summary['n_entities']} "
        f"relations={summary['n_relations']} "
        f"({summary['build_seconds']}s)"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true",
                        help="delete existing fixture before building")
    parser.add_argument("--device", default=None,
                        help="torch device (default: cuda if available)")
    args = parser.parse_args(argv)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    summaries = []
    for entry in KNOWLEDGE:
        summaries.append(_build_one(entry, device, rebuild=args.rebuild))

    manifest = {
        "build_seed": BUILD_SEED,
        "build_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ingest_epochs": INGEST_EPOCHS,
        "geometry_m": GEOMETRY_M,
        "device": str(device),
        "stores": summaries,
    }
    manifest_path = FIXTURES_DIR / "MANIFEST.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                prior = json.load(f)
        except json.JSONDecodeError:
            prior = {"stores": []}
        prior_by_name = {s.get("name"): s for s in prior.get("stores", [])}
        for i, s in enumerate(summaries):
            if s.get("skipped") and s["name"] in prior_by_name:
                summaries[i] = prior_by_name[s["name"]]
        manifest["stores"] = summaries
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[manifest] {manifest_path}")

    print("\n=== summary ===")
    for s in summaries:
        if s.get("skipped"):
            print(f"  {s['name']}: skipped (existing)")
        else:
            print(
                f"  {s['name']}: mode={s['ingest_mode']} "
                f"triples={s['n_triples']} enm={s['n_enm']} "
                f"entities={s['n_entities']} relations={s['n_relations']} "
                f"in {s['build_seconds']}s"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
