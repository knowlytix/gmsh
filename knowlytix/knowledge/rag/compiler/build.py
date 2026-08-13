# SPDX-License-Identifier: Apache-2.0
"""One-call build of a query compiler paired with a GMS store.

:func:`build_query_compiler` runs the full pipeline — generate DoE-varied training
data from the store, LoRA-fine-tune the compiler, and save the adapter into
``store_dir/query_compiler/`` — so that a subsequent
``RagConfig.from_store_dir(store_dir, llm)`` automatically routes queries through the
compiler. This mirrors :func:`~knowlytix.knowledge.geode.rag.build_calibrated_rag_store`:
artifacts live in the store directory and are auto-wired on read.
"""

from __future__ import annotations

import json
import os

from .config import COMPILER_SUBDIR, CompilerSFTConfig
from .datagen import build_compiler_dataset
from .train import train_compiler

__all__ = ["build_query_compiler"]


def _load_store(store_dir: str, device: str):
    from knowlytix.knowledge.config import DocGMSConfig
    from knowlytix.knowledge.store import GMSExpertStore

    store = GMSExpertStore(DocGMSConfig(store_path=store_dir), device=device)
    store.load()
    return store


def build_query_compiler(
    store_dir: str,
    llm,
    *,
    store=None,
    sft_config: CompilerSFTConfig | None = None,
    group: str = "comprehensive",
    variants_per_base: int = 12,
    max_multi_hop: int = 150,
    negatives_per_head: int = 1,
    heldout_levels: dict | None = None,
    heldout_fact_frac: float = 0.0,
    device: str = "cuda",
    evaluate: bool = True,
    seed: int = 42,
) -> dict:
    """Build and persist a query compiler for the store at ``store_dir``.

    Args:
        store_dir: the persisted GMS store directory; the adapter is written to
            ``store_dir/query_compiler/`` and the data to ``.../query_compiler/data/``.
        llm: an :class:`LLMBackend` used to rephrase the training questions
            (e.g. ``LocalTransformersBackend(QWEN_4B)``).
        store: a preloaded store (loaded from ``store_dir`` if omitted).
        sft_config: LoRA hyperparameters (default :class:`CompilerSFTConfig`).
        group, variants_per_base, max_multi_hop, negatives_per_head, heldout_levels,
        heldout_fact_frac, seed: passed to
            :func:`~knowlytix.knowledge.rag.compiler.datagen.build_compiler_dataset`.
        evaluate: score the trained compiler on the store's accept cohort and
            attach the metrics to the manifest.

    Returns:
        A manifest dict: ``{compiler_dir, base_model, n_train, n_heldout_level,
        n_heldout_fact, factors[, metrics]}``. Also written to
        ``store_dir/query_compiler/compiler_meta.json``.

    Memory: the three phases (datagen, train, eval) each load a Qwen-sized base
    model, but only one is resident at a time — the rephraser is released after
    datagen and the SFT model after training — so peak GPU memory is one model,
    not three. This is what keeps the call from OOM-ing the GB10's unified memory.
    The passed ``llm`` is unloaded after datagen and must not be reused afterward.
    """
    cfg = sft_config or CompilerSFTConfig()
    if store is None:
        store = _load_store(store_dir, device)

    comp_dir = os.path.join(store_dir, COMPILER_SUBDIR)
    data_dir = os.path.join(comp_dir, "data")
    os.makedirs(comp_dir, exist_ok=True)

    splits = build_compiler_dataset(
        store, llm, group=group, variants_per_base=variants_per_base,
        max_multi_hop=max_multi_hop, negatives_per_head=negatives_per_head,
        heldout_levels=heldout_levels, heldout_fact_frac=heldout_fact_frac,
        seed=seed, out_dir=data_dir)

    # Datagen is the only phase that uses the rephraser LLM; release it so training
    # and evaluation each hold a single Qwen copy rather than stacking a second (train)
    # and third (eval) on top of it — the unified-memory OOM this pipeline used to hit,
    # delivered as SIGKILL before any log could flush. unload() is a no-op for hosted
    # backends and for any backend without on-device weights. evaluate_compiler never
    # calls this llm (the accept cohort runs with generate off), so freeing it is safe.
    getattr(llm, "unload", lambda: None)()

    train_compiler(splits["train"], comp_dir, config=cfg)

    manifest = {
        "compiler_dir": comp_dir,
        "base_model": cfg.base_model,
        "n_train": len(splits["train"]),
        "n_heldout_level": len(splits["test_heldout_level"]),
        "n_heldout_fact": len(splits["test_heldout_fact"]),
        "factor_group": group,
        "variants_per_base": variants_per_base,
    }

    if evaluate:
        from .evaluate import evaluate_compiler
        manifest["metrics"] = evaluate_compiler(
            store, llm, compiler_dir=comp_dir, device=device,
            heldout=splits["test_heldout_level"] or None, seed=seed)

    with open(os.path.join(comp_dir, "compiler_meta.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
