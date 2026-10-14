# SPDX-License-Identifier: Apache-2.0
"""The trained query compiler as a RAG query front-end.

:class:`CompiledQueryParser` loads a base causal LM plus the LoRA adapter trained
by :mod:`~knowlytix.knowledge.rag.compiler.train`, and exposes ``extract`` — the
same contract as ``GeometricQueryParser`` / ``QueryTripleExtractor`` — so it drops
into :class:`~knowlytix.knowledge.rag.pipeline.RagPipeline` via
``query_parse_mode="compiler"``.

The compiler emits a HOP CHAIN (JSON array of ``{head, relation}``) for the entity
the question NAMES. The parser then walks that chain against the store by EXACT
lookup, hop by hop: the first head is resolved to a store entity, each hop must be
an asserted fact, and the intermediate entity is followed to the next hop. A chain
that breaks at any hop (a relation the current entity does not hold) yields the
empty list, so the pipeline abstains through its existing no-binding path — it never
drifts to a different head that happens to hold the relation. A valid chain is
returned as ``?x``-chained :class:`QueryTriple`s (variables for the intermediate and
asked slots), the same shape the geometric parser produces, so the pipeline's
multi-hop retrieval resolves it and returns provenance-linked facts.
"""

from __future__ import annotations

from knowlytix.knowledge.rag.query_triples import QueryTriple

from .config import COMPILER_SYSTEM, parse_target
from .walk import STRUCTURAL_RELATIONS as _STRUCTURAL
from .walk import StoreChainWalker

__all__ = ["CompiledQueryParser"]


class CompiledQueryParser:
    """NL question -> exact-walked ``(head, relation, ?)`` query-triple chain.

    Args:
        store: a loaded :class:`GMSExpertStore`.
        adapter_dir: directory holding the trained LoRA adapter + tokenizer.
        base_model: HuggingFace id of the base model the adapter applies to. If
            ``None``, read from the adapter's ``adapter_config.json``.
        device: torch device for inference.
        max_new_tokens: generation budget for the JSON output (chains need room).
        structural_relations: relation ids treated as structural graph scaffolding
            and excluded from the walkable content facts. Defaults to
            :data:`_STRUCTURAL` (the store-builder's conventional structural edges);
            pass a store's own set to keep the compiler store-agnostic.
    """

    def __init__(self, store, adapter_dir: str, *, base_model: str | None = None,
                 device: str = "cuda", max_new_tokens: int = 96,
                 structural_relations: "set[str] | None" = None):
        self.store = store
        self.adapter_dir = adapter_dir
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._structural = (set(_STRUCTURAL) if structural_relations is None
                            else set(structural_relations))
        self._walker = StoreChainWalker(store, structural_relations=self._structural)
        self._load_model(adapter_dir, base_model)

    def _walk_chain(self, hops: list[tuple[str, str]]) -> list[QueryTriple]:
        """Walk an emitted hop chain against the store; see
        :meth:`~knowlytix.knowledge.rag.compiler.walk.StoreChainWalker.walk`."""
        return self._walker.walk(hops)

    # -- model -------------------------------------------------------------

    def _load_model(self, adapter_dir: str, base_model: str | None) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The query compiler needs the inference stack (transformers, peft). "
                'Install with:  pip install "knowlytix-knowledge[compiler]"'
                f"  (missing: {e.name})") from e
        if base_model is None:
            import json
            import os
            cfg_path = os.path.join(adapter_dir, "adapter_config.json")
            base_model = json.loads(open(cfg_path).read())["base_model_name_or_path"]
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(adapter_dir)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=torch.bfloat16, device_map=self.device)
        self.model = PeftModel.from_pretrained(model, adapter_dir)
        self.model.eval()

    def _generate(self, questions: list[str], chunk: int = 32) -> list[str]:
        torch = self._torch
        outs: list[str] = []
        for i in range(0, len(questions), chunk):
            batch = questions[i:i + chunk]
            texts = [self.tok.apply_chat_template(
                [{"role": "system", "content": COMPILER_SYSTEM},
                 {"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True) for q in batch]
            enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                    pad_token_id=self.tok.pad_token_id)
            for j in range(len(batch)):
                new = gen[j][enc["input_ids"].shape[1]:]
                outs.append(self.tok.decode(new, skip_special_tokens=True))
        return outs

    # -- raw compilation (before store resolution) -------------------------

    def compile_raw(self, questions: list[str]) -> list[list[tuple[str, str]]]:
        """The model's raw hop chain ``[(head, relation), ...]`` for each question,
        BEFORE the store walk. Use it to measure generation fidelity separately from
        the store's abstention. ``[]`` for an unparseable generation."""
        return [parse_target(out) for out in self._generate(list(questions))]

    # -- extractor contract ------------------------------------------------

    def extract(self, question: str, *, hint: str | None = None) -> list[QueryTriple]:
        """Compile ``question`` to a ``?x``-chained query-triple list, or ``[]`` when
        the emitted chain is not fully asserted in the store (abstain).

        ``hint`` is accepted for parity with the extractor contract and is currently
        unused: the compiler conditions only on the question text."""
        return self._walk_chain(self.compile_raw([question])[0])

    def extract_batch(self, questions: list[str]) -> list[list[QueryTriple]]:
        """Batched :meth:`extract` — one padded forward pass per chunk."""
        return [self._walk_chain(chain) for chain in self.compile_raw(questions)]
