# SPDX-License-Identifier: Apache-2.0
"""The two build-new bake-off binders: exemplar few-shot (Arm C) and the
alias-table hybrid (Arm D).

Both are *query front-ends* -- they implement the extractor contract
(``extract(question) -> list[QueryTriple]``, and a batched ``extract_batch``) that
:class:`~knowlytix.knowledge.rag.pipeline.RagPipeline` selects as ``self.extractor``,
so a caller drops either in by assigning ``pipe.extractor = <parser>`` after building
an otherwise-standard pipeline. Every stage after parsing -- retrieval, the
relevance gate, verification, assembly, ENM, synthesis -- is then shared with arms A
and B, so a measured delta is the binder's.

Both emit a ``(head, relation)`` hop chain and resolve it with the compiler's
:class:`~knowlytix.knowledge.rag.compiler.walk.StoreChainWalker`, so their
abstention semantics match arm B exactly (a hop that is not an asserted fact yields
the empty chain, and the pipeline abstains through its no-binding path). They differ
only in how the chain is produced:

* **Arm C -- exemplar few-shot.** A FROZEN small LM (no adapter) is prompted with the
  ``k`` training exemplars nearest the question in the tuned encoder's space, each
  shown as ``question -> hop-chain``. Patching a mis-binding edits the exemplar pool,
  a config change (``patch_mechanism="alias_edit"``); nothing is trained.
* **Arm D -- alias-table hybrid.** The trained compiler supplies the *relations*
  (the learned part), but the *head entity* is resolved by a patchable alias table
  (surface form -> canonical id) with a calibrated encoder-nearest fallback. A novel
  or mis-resolved entity is fixed by adding one alias-table row -- an
  ``alias_edit`` -- rather than retraining the compiler.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The alias-table hybrid (Arm D) is now a product binder in the compiler
# subpackage (query_parse_mode="alias_hybrid"); re-exported here so existing
# ``from ...rag.bakeoff import AliasHybridParser`` imports keep working.
from knowlytix.knowledge.rag.compiler.alias_hybrid import (  # noqa: F401
    AliasHybridParser,
    AliasTableResolver,
    build_alias_hybrid_parser,
    calibrate_alias_resolver,
)
from knowlytix.knowledge.rag.compiler.config import (
    COMPILER_SYSTEM,
    format_target,
    parse_target,
)
from knowlytix.knowledge.rag.compiler.walk import StoreChainWalker

__all__ = [
    "ExemplarIndex",
    "ExemplarFewShotParser",
    "AliasTableResolver",
    "AliasHybridParser",
    "build_alias_hybrid_parser",
    "calibrate_alias_resolver",
]


def _cos_topk(query_vec, mat, k):
    """Indices of the ``k`` rows of ``mat`` most cosine-similar to ``query_vec``.
    Both are assumed L2-normalized; ties break by index."""
    import numpy as np

    sims = mat @ query_vec
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k - 1)[:k]
    return idx[np.argsort(-sims[idx])]


def _normalize_rows(mat):
    import numpy as np

    mat = np.asarray(mat, dtype="float32")
    if mat.ndim == 1:
        mat = mat[None, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# ------------------------------------------------------------------ Arm C ----

@dataclass
class ExemplarIndex:
    """A k-NN index over training exemplars in the tuned encoder's space.

    Each exemplar is a ``(question, hops)`` pair; ``hops`` is the gold
    ``[(head, relation), ...]`` chain. :meth:`nearest` returns the ``k`` exemplars
    closest to a query question, for dynamic few-shot prompting.
    """

    questions: list[str]
    hops: list[list]
    _emb: object = field(default=None, repr=False)

    @classmethod
    def from_rows(cls, rows: list[dict], encoder) -> "ExemplarIndex":
        """Build from ``llm_extract_sft`` rows ``{"nl": ..., "triples": [{head,
        relation,...}]}``. Rows without a usable chain are dropped."""
        qs, hp = [], []
        for r in rows:
            q = r.get("nl") or r.get("question")
            trips = r.get("triples") or r.get("hops") or []
            chain = []
            for t in trips:
                if isinstance(t, dict):
                    h, rel = t.get("head"), t.get("relation")
                elif isinstance(t, (list, tuple)) and len(t) >= 2:
                    h, rel = t[0], t[1]
                else:
                    h = rel = None
                if isinstance(h, str) and isinstance(rel, str):
                    chain.append((h, rel))
            if q and chain:
                qs.append(q)
                hp.append(chain)
        emb = _normalize_rows(encoder(qs)) if qs else None
        return cls(questions=qs, hops=hp, _emb=emb)

    def __len__(self) -> int:
        return len(self.questions)

    def nearest(self, query_vec, k: int) -> list[tuple[str, list]]:
        if not self.questions:
            return []
        qv = _normalize_rows(query_vec)[0]
        return [(self.questions[i], self.hops[i]) for i in _cos_topk(qv, self._emb, k)]


class ExemplarFewShotParser:
    """Arm C: a frozen small LM prompted with the nearest training exemplars.

    Args:
        store: a loaded GMS store (supplies the chain walk).
        backend: an ``LLMBackend`` with ``call_batch(users, system, max_tokens)`` --
            the FROZEN base model (no adapter). Reuse the one already loaded for the
            arm-A extractor to avoid a second model copy.
        index: an :class:`ExemplarIndex` over the training exemplars.
        encoder: the tuned v-encoder (``list[str] -> array``) used to embed queries.
        k: exemplars per prompt.
        structural_relations: passed through to the walker.
        max_new_tokens: generation budget for the JSON chain.
    """

    def __init__(self, store, backend, index: ExemplarIndex, encoder, *, k: int = 8,
                 structural_relations=None, max_new_tokens: int = 96,
                 batch_chunk: int = 32):
        self.store = store
        self.backend = backend
        self.index = index
        self.encoder = encoder
        self.k = k
        self.max_new_tokens = max_new_tokens
        self.batch_chunk = batch_chunk
        self._walker = StoreChainWalker(store, structural_relations=structural_relations)

    def _prompt_for(self, question: str, demos: list[tuple[str, list]]) -> str:
        lines = ["Examples:"]
        for q, hops in demos:
            lines.append(f"Q: {q}")
            lines.append(f"A: {format_target(hops)}")
        lines.append("")
        lines.append(f"Q: {question}")
        lines.append("A:")
        return "\n".join(lines)

    def compile_raw(self, questions: list[str]) -> list[list[tuple[str, str]]]:
        """The frozen model's raw hop chains, BEFORE the store walk. Generation runs
        in ``batch_chunk``-sized padded passes so a large cohort does not build one
        oversized batch (the few-shot prompts are long; a single pass OOMs)."""
        questions = list(questions)
        if not questions:
            return []
        vecs = _normalize_rows(self.encoder(questions))
        users = [self._prompt_for(q, self.index.nearest(vecs[i], self.k))
                 for i, q in enumerate(questions)]
        outs: list[str] = []
        for i in range(0, len(users), self.batch_chunk):
            outs.extend(self.backend.call_batch(
                users[i:i + self.batch_chunk], system=COMPILER_SYSTEM,
                max_tokens=self.max_new_tokens))
        return [parse_target(o) for o in outs]

    def extract(self, question: str, *, hint: str | None = None):
        return self._walker.walk(self.compile_raw([question])[0])

    def extract_batch(self, questions: list[str]):
        return [self._walker.walk(chain) for chain in self.compile_raw(list(questions))]
