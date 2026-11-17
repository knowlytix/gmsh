# SPDX-License-Identifier: Apache-2.0
"""Shared configuration and constants for the LLM query compiler.

The compiler is a small causal LM fine-tuned to map a natural-language question
to the ``(head, relation)`` of the NAMED entity for a fixed GMS store. The system
prompt and target format defined here are the single source of truth shared by
data generation (label format), training (SFT target), and inference
(:class:`~knowlytix.knowledge.rag.compiler.model.CompiledQueryParser`) — they must
match exactly or the trained model and the parser disagree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

# Default base model: the trilogy/GEODE-native small instruct model.
from knowlytix.knowledge.geode.agent_llm import QWEN_4B

__all__ = [
    "COMPILER_SUBDIR",
    "COMPILER_SYSTEM",
    "QWEN_4B",
    "format_target",
    "parse_target",
    "CompilerSFTConfig",
]

# Store subdirectory holding the trained adapter + metadata, mirroring the
# tuned_encoder/ contradiction_encoder/ convention of build_calibrated_rag_store.
COMPILER_SUBDIR = "query_compiler"

# The compiler emits a HOP CHAIN: a JSON array of {"head","relation"} objects, one
# per hop. A single-hop question is a one-element array; a multi-hop question chains
# through intermediate entities, whose head slots are variables ("?x0", "?x1", ...).
# The chain is ALWAYS the faithful path from the entity the question NAMES, for
# answerable AND relation-absent questions alike. Abstention is decided downstream
# by the store walk (a hop that is not an asserted fact yields no retrieval, so the
# pipeline abstains). Faithfulness to the named head is the whole objective: the
# model must never substitute a different, more familiar entity or attribute.
COMPILER_SYSTEM = (
    "You are a query compiler for a fixed knowledge store. Given a question, output "
    "the chain of (head, relation) hops it asks about, as a JSON array: "
    '[{"head": <id>, "relation": <id>}, ...]. A single-hop question is one object; '
    'a multi-hop question chains hops, with intermediate heads written as "?x0", '
    '"?x1", ... Use the exact entity named in the question as the first head; never '
    "substitute a different entity."
)


def format_target(hops: list[tuple[str, str]]) -> str:
    """The SFT target string for a hop chain ``[(head, relation), ...]``: a compact
    JSON array with stable key order. A single-hop chain is a one-element array."""
    return json.dumps([{"head": h, "relation": r} for h, r in hops])


def parse_target(text: str) -> list[tuple[str, str]]:
    """Parse a model generation into a hop chain ``[(head, relation), ...]``; ``[]``
    on failure. Accepts a JSON array of objects (preferred) or a single bare object
    (tolerant of a lone-hop generation), ignoring surrounding prose or code fences."""
    import re

    def _hops(obj):
        items = obj if isinstance(obj, list) else [obj]
        out = []
        for d in items:
            if not isinstance(d, dict):
                return []
            h, r = d.get("head"), d.get("relation")
            if not isinstance(h, str) or not isinstance(r, str):
                return []
            out.append((h, r))
        return out

    m = re.search(r"\[.*\]", text, re.DOTALL)          # prefer a JSON array
    if m:
        try:
            return _hops(json.loads(m.group(0)))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*?\}", text, re.DOTALL)          # fall back to a single object
    if m:
        try:
            return _hops(json.loads(m.group(0)))
        except json.JSONDecodeError:
            return []
    return []


@dataclass
class CompilerSFTConfig:
    """Hyperparameters for LoRA supervised fine-tuning of the query compiler.

    Defaults reproduce the validated CIP run (r16/a32 over all linear layers,
    3 epochs, assistant-only loss, bf16). ``base_model`` is the HuggingFace id of
    the model to adapt.
    """

    base_model: str = QWEN_4B
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"
    # Optimization
    epochs: int = 3
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    # Effective batch is per_device_batch_size * grad_accum_steps = 16. Keep the
    # per-device batch small: the GB10 has unified memory (GPU allocations come out
    # of the same pool as the OS), so a large per-device batch of activations can
    # exhaust system RAM and hard-hang the machine rather than raise a CUDA OOM.
    per_device_batch_size: int = 2
    grad_accum_steps: int = 8
    max_length: int = 256
    bf16: bool = True
    # Trade compute for memory: recompute activations in the backward pass instead
    # of holding them all resident. Needed on the GB10's unified memory.
    gradient_checkpointing: bool = True
    seed: int = 42
    device: str = "cuda"
