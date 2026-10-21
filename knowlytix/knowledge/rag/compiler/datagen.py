# SPDX-License-Identifier: Apache-2.0
"""Training-data generation for the LLM query compiler — a thin caller of the
knowlytix question-generation stack (no hand-rolled templates or rephrasing).

Pipeline, each stage bound to a canonical knowlytix API:

  1. BASE questions  -> knowlytix.benchmark.generators mining ``store.doc_graph``:
       * FactRecallGenerator        — single-hop (head, relation) facts (reused);
       * RelationChainGenerator      — 2-hop relation chains (hop_count = 2-Hop);
       * RelationAbsentProbeGenerator — negative probes (teach faithful abstention).
     Each GeneratedQuestion carries the fact/chain in ``metadata`` (head/relation or
     hops), which is the compiler target and is held INVARIANT through rephrasing.
  2. DoE design      -> DesignMatrix.from_catalog(group=...) — a named factor group
     from the graphdoe factor_catalog (default "comprehensive", 20 factors).
  3. Rephrasing      -> QuestionRephraser(method="llm").expand(base, matrix) — the
     canonical rephraser, run on a local model via LocalTransformersLLMClient. The
     ground truth (the target chain) is preserved by the rephraser.

The only compiler-specific step here is reading each expanded question's target
chain out of its preserved metadata and writing the train/held-out split. Everything
that generates or varies a question is a knowlytix component; see the stage bindings
above. A held-out split reserves one presentation LEVEL (abbreviated aliasing) to
measure generalization to an unseen surface form.

Chain depth is 1 (FactRecallGenerator) and 2 (RelationChainGenerator). The target
format and the parser walk handle chains of any depth, but no generator here emits
chains deeper than 2, so questions requiring three or more hops are out of the
trained distribution.
"""

from __future__ import annotations

import json
import os

__all__ = ["build_compiler_dataset", "DEFAULT_GROUP", "DEFAULT_HELDOUT_LEVELS"]

DEFAULT_GROUP = "comprehensive"
DEFAULT_HELDOUT_LEVELS = {"entity_aliasing": "Abbreviated"}


def _as_client(llm, *, batch_chunk=8):
    """Return an LLMClient-compatible object for QuestionRephraser. Accepts an
    LLMClient (has ``complete_batch``) or a LocalTransformersBackend (wrapped).

    ``batch_chunk`` bounds how many rephrase prompts a single ``generate()`` call
    submits. On the GB10's unified pool a large chunk (128) is a one-shot allocation
    spike that thrashes swap and freezes the node before a coarse memory watchdog can
    fire, so the default is kept small; raise it only on hardware with headroom.
    """
    if hasattr(llm, "complete_batch"):
        return llm
    from knowlytix.knowledge.llm_backend import LocalTransformersLLMClient
    # Cap rephrase generation: a rewritten question is short (the rephraser rejects
    # >400 chars), so bound greedy decoding well below expand's 1024 ceiling.
    return LocalTransformersLLMClient(getattr(llm, "model_name", ""), backend=llm,
                                      batch_chunk=batch_chunk, max_tokens_cap=128)


def build_compiler_dataset(
    store,
    llm,
    *,
    group: str = DEFAULT_GROUP,
    factor_names: list[str] | None = None,
    variants_per_base: int = 12,
    max_multi_hop: int = 150,
    negatives_per_head: int = 1,
    heldout_levels: dict | None = None,
    heldout_fact_frac: float = 0.0,
    seed: int = 42,
    out_dir: str | None = None,
    rephrase_batch_chunk: int = 8,
) -> dict[str, list[dict]]:
    """Generate rich, catalog-driven ``question -> (head, relation) chain`` data.

    Args:
        store: a loaded :class:`GMSExpertStore` (needs ``doc_graph``).
        llm: a ``core.llm.LLMClient`` or a ``LocalTransformersBackend`` (wrapped in
            :class:`LocalTransformersLLMClient`) used by ``QuestionRephraser``.
        group: graphdoe factor-group name for the DoE (default "comprehensive").
        factor_names: explicit factor subset for the DoE; when given, overrides
            ``group`` (used to hold factors out of training for an OOD holdout).
        variants_per_base: DoE runs (= presentation variants) per base question.
        max_multi_hop: cap on 2-hop base chains (0 = all).
        negatives_per_head: relation-absent probes per head.
        heldout_levels: ``{factor: level}`` routed to the held-out-level split
            (default abbreviated aliasing).
        heldout_fact_frac: fraction of entities reserved as unseen facts.
        out_dir: if given, writes the three jsonl splits there.
        rephrase_batch_chunk: prompts per rephrase ``generate()`` call; small by
            default to avoid a one-shot memory spike on unified-memory GPUs.

    Returns:
        ``{"train", "test_heldout_level", "test_heldout_fact"}``; each row has
        ``question, hops, head, relation, category, factors, heldout_level,
        heldout_fact``.
    """
    import random

    from knowlytix.benchmark.generators import (
        FactRecallGenerator,
        RelationAbsentProbeGenerator,
        RelationChainGenerator,
    )
    from knowlytix.harness.graphdoe import DesignMatrix, QuestionRephraser

    heldout_levels = DEFAULT_HELDOUT_LEVELS if heldout_levels is None else heldout_levels
    graph = store.doc_graph
    client = _as_client(llm, batch_chunk=rephrase_batch_chunk)

    # 1. base questions from the knowlytix generators
    base = (FactRecallGenerator().generate(graph)
            + RelationChainGenerator(max_chains=max_multi_hop, seed=seed).generate(graph)
            + RelationAbsentProbeGenerator(negatives_per_head=negatives_per_head,
                                           seed=seed).generate(graph))

    # 2. DoE design over a named factor group, or an explicit factor subset when
    #    factor_names is given (e.g. to hold out a set of factors for an
    #    out-of-distribution evaluation). factor_names takes precedence over group.
    dm = DesignMatrix.from_catalog(
        group=None if factor_names else group, factor_names=factor_names,
        n_runs=variants_per_base, method="sobol", seed=seed)
    matrix = dm.generate()

    # 3. canonical rephrase (local model), ground truth preserved
    doe = QuestionRephraser(method="llm", llm=client).expand(base, matrix)

    heads = sorted({(q.question.metadata.get("head") or "") for q in doe})
    hf_rng = random.Random(seed + 7)
    n_heldout = int(heldout_fact_frac * len(heads))
    heldout_facts = set(hf_rng.sample(heads, n_heldout)) if n_heldout else set()

    rows: list[dict] = []
    for dq in doe:
        md = dq.question.metadata
        hops = md.get("hops") or [[md.get("head"), md.get("relation")]]
        h0 = hops[0][0]
        fa = dict(dq.factor_assignment)
        rows.append({
            "question": dq.question.natural_language,
            "hops": [[h, r] for h, r in hops],
            "head": h0, "relation": hops[0][1],
            "category": dq.question.category,
            "factors": fa,
            "heldout_level": any(fa.get(f) == lv for f, lv in heldout_levels.items()),
            "heldout_fact": h0 in heldout_facts,
        })

    split: dict[str, list[dict]] = {"train": [], "test_heldout_level": [],
                                    "test_heldout_fact": []}
    for row in rows:
        if row["heldout_fact"]:
            split["test_heldout_fact"].append(row)
        elif row["heldout_level"]:
            split["test_heldout_level"].append(row)
        else:
            split["train"].append(row)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for name, part in split.items():
            with open(os.path.join(out_dir, f"{name}.jsonl"), "w") as f:
                for row in part:
                    f.write(json.dumps(row) + "\n")
    return split
