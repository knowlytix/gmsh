# SPDX-License-Identifier: Apache-2.0
"""LLM query compiler: a store-specific, LoRA-fine-tuned small LM query front-end.

Pairs a GMS store with a trained compiler that maps a natural-language question to
the NAMED entity's ``(head, relation)``, resolved against the store by exact lookup.
Used as the ``query_parse_mode="compiler"`` front-end of
:class:`~knowlytix.knowledge.rag.pipeline.RagPipeline`, it removes the
peripheral-head false-accept of geometric binding: a relation the named head does
not hold resolves to no fact and abstains, rather than drifting to a different head.

Typical use::

    from knowlytix.knowledge.rag.compiler import build_query_compiler
    from knowlytix.knowledge.rag import RagConfig, RagPipeline

    build_query_compiler(store_dir, llm)              # datagen -> LoRA SFT -> save
    rag  = RagConfig.from_store_dir(store_dir, llm)   # auto-detects the compiler
    pipe = RagPipeline.from_store(store, rag)
    ans  = pipe.query("What is the CIP identity-verification requirement?")

The training stack (transformers/peft/trl/datasets) is the optional ``compiler``
extra: ``pip install "knowlytix[compiler]"``.
"""

from __future__ import annotations

from .build import build_query_compiler
from .config import (
                     COMPILER_SUBDIR,
                     COMPILER_SYSTEM,
                     QWEN_4B,
                     CompilerSFTConfig,
                     format_target,
                     parse_target,
)
from .datagen import build_compiler_dataset
from .evaluate import GEOMETRIC_BASELINE_CIP, evaluate_compiler
from .model import CompiledQueryParser
from .train import train_compiler
from .walk import StoreChainWalker

__all__ = [
    "build_query_compiler",
    "build_compiler_dataset",
    "train_compiler",
    "CompiledQueryParser",
    "StoreChainWalker",
    "evaluate_compiler",
    "GEOMETRIC_BASELINE_CIP",
    "CompilerSFTConfig",
    "COMPILER_SUBDIR",
    "COMPILER_SYSTEM",
    "QWEN_4B",
    "format_target",
    "parse_target",
]
