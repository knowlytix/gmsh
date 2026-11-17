# SPDX-License-Identifier: Apache-2.0
"""LoRA supervised fine-tuning of the query compiler.

Trains a small causal LM to emit ``{"head": ..., "relation": ...}`` for a question,
using the data produced by :mod:`~knowlytix.knowledge.rag.compiler.datagen`. The SFT
stack (transformers/peft/trl/datasets) is an optional dependency group; it is
imported lazily so that importing knowlytix never requires it. Install with::

    pip install "knowlytix-knowledge[compiler]"
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import COMPILER_SYSTEM, CompilerSFTConfig, format_target

__all__ = ["train_compiler"]

_INSTALL_HINT = (
    "The query compiler's training stack is not installed. Install the optional "
    'dependencies with:  pip install "knowlytix-knowledge[compiler]"  '
    "(transformers, peft, trl, datasets, accelerate)."
)


def _require_sft_stack():
    """Import the SFT stack, raising a clear, actionable error if it is absent."""
    try:
        import torch  # noqa: F401
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(f"{_INSTALL_HINT}\n(missing: {e.name})") from e
    return (Dataset, LoraConfig, AutoModelForCausalLM, AutoTokenizer,
            SFTConfig, SFTTrainer)


def _load_rows(data) -> list[dict]:
    """Accept a jsonl path (str/Path) or an in-memory list of row dicts."""
    if isinstance(data, (str, Path)):
        return [json.loads(line) for line in open(data)]
    return list(data)


def _to_messages(rows: list[dict]):
    from datasets import Dataset

    contracts = {r.get("compiler_contract", "legacy") for r in rows}
    if len(contracts) != 1:
        raise ValueError("Cannot mix compiler input/output contracts")
    msgs = []
    for r in rows:
        if "compiler_contract" in r:
            from knowlytix.discovery.dynaquery.compiler.schema_contract import (
                training_messages,
            )
            msgs.append({"messages": training_messages(r)})
            continue
        hops = [tuple(x) for x in r["hops"]] if "hops" in r else [(r["head"], r["relation"])]
        msgs.append({"messages": [
            {"role": "system", "content": COMPILER_SYSTEM},
            {"role": "user", "content": r["question"]},
            {"role": "assistant", "content": format_target(hops)},
        ]})
    return Dataset.from_list(msgs)


def train_compiler(
    data,
    out_dir: str,
    *,
    config: CompilerSFTConfig | None = None,
    smoke: bool = False,
) -> str:
    """LoRA-fine-tune a query compiler and save the adapter to ``out_dir``.

    Args:
        data: training rows — a path to a ``train.jsonl`` (from
            :func:`~knowlytix.knowledge.rag.compiler.datagen.build_compiler_dataset`)
            or an in-memory list of ``{"question","head","relation",...}`` dicts.
        out_dir: directory to write the LoRA adapter + tokenizer into.
        config: :class:`CompilerSFTConfig` hyperparameters (defaults reproduce the
            validated run).
        smoke: run 5 steps only, do not save — a fast pipeline check.

    Returns:
        ``out_dir`` (the saved adapter path), or ``""`` for a smoke run.
    """
    cfg = config or CompilerSFTConfig()
    (Dataset, LoraConfig, AutoModelForCausalLM, AutoTokenizer,
     SFTConfig, SFTTrainer) = _require_sft_stack()
    import torch

    rows = _load_rows(data)
    if not rows:
        raise ValueError("no training rows")
    ds = _to_messages(rows)

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if rows[0].get("compiler_contract"):
        # Schema input is longer than legacy question-only input. Never silently
        # truncate it or the supervised query target.
        for messages in ds["messages"]:
            tokens = tok.apply_chat_template(messages, tokenize=True)
            ids = tokens["input_ids"] if hasattr(tokens, "keys") else tokens
            if len(ids) > cfg.max_length:
                raise ValueError("Compiler schema/target exceeds max_length; increase the limit")
    dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, dtype=dtype, device_map=cfg.device)
    peft_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=cfg.target_modules)

    sampler_options = {}
    if rows[0].get("compiler_contract"):
        # Schema sizes vary substantially. Batch similar lengths to avoid padding
        # every short request to the largest schema in a randomly mixed batch.
        fields = SFTConfig.__dataclass_fields__
        if "train_sampling_strategy" in fields:
            sampler_options["train_sampling_strategy"] = "group_by_length"
        elif "group_by_length" in fields:
            sampler_options["group_by_length"] = True
    sft = SFTConfig(
        **sampler_options,
        output_dir=out_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler,
        warmup_ratio=cfg.warmup_ratio,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        max_length=cfg.max_length,
        packing=False,
        max_steps=5 if smoke else -1,
        assistant_only_loss=True,
        seed=cfg.seed,
    )
    trainer = SFTTrainer(model=model, args=sft, train_dataset=ds,
                         processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    try:
        if smoke:
            return ""
        trainer.save_model(out_dir)
        tok.save_pretrained(out_dir)
        if rows[0].get("compiler_contract"):
            import hashlib

            from knowlytix.discovery.dynaquery.compiler.schema_contract import (
                CONTRACT_VERSION,
                SCHEMA_SYSTEM,
            )
            (Path(out_dir) / "compiler_contract.json").write_text(json.dumps({
                "version": CONTRACT_VERSION,
                "system_sha256": hashlib.sha256(SCHEMA_SYSTEM.encode()).hexdigest(),
                "query_source": "question and live schema", "answers_in_input": False,
            }, indent=2) + "\n")
        return out_dir
    finally:
        # Release the SFT model and empty the CUDA allocator before returning, so a
        # following phase (evaluation) loads a single Qwen copy rather than stacking
        # one on top of the trainer's. Left to interpreter GC, the model co-resides
        # with the eval model and can OOM the GB10's unified memory.
        import gc

        try:
            del trainer, model
        except NameError:  # pragma: no cover
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
