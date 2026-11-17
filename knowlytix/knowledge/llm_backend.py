# SPDX-License-Identifier: Apache-2.0
"""Unified LLM interface for the ``knowlytix.knowledge`` package.

Two backend flavors are exposed:

- ``LiteLLMBackend`` — default for hosted providers (Anthropic, OpenAI,
  Google, Ollama, …). Routes through :mod:`gms.llm`, which wraps LiteLLM,
  so model selection, timeouts, retries, and temperature come from
  :class:`gms.llm.LLMSettings` (``GMS_LLM_*`` env vars).
- ``LocalTransformersBackend`` — unchanged, for on-device HuggingFace
  models on GPU.

For backwards compatibility, ``AnthropicBackend`` is retained as a thin
subclass of ``LiteLLMBackend`` that auto-prepends the ``anthropic/``
provider prefix when the caller passes a bare Claude model name (e.g.
``"claude-sonnet-4-20250514"``). New code should prefer
``LiteLLMBackend`` directly.
"""

from abc import ABC, abstractmethod

from knowlytix.knowledge.config import ConvertConfig


class LLMBackend(ABC):
    """Abstract LLM backend."""

    @abstractmethod
    def call(self, system: str, user: str, max_tokens: int = 2048) -> str:
        """Complete a single ``system``/``user`` prompt pair and return the text."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    def call_batch(self, users: "list[str]", system: str = "",
                   max_tokens: int = 2048) -> "list[str]":
        """Complete many user prompts and return the texts in order. The default
        is a sequential loop over :meth:`call`; backends with a native batch path
        (e.g. a local model's single padded forward pass,
        :class:`LocalTransformersBackend`) override this. Callers can therefore
        batch against any backend and get the fast path where one exists."""
        return [self.call(system, u, max_tokens) for u in users]

    def unload(self) -> None:
        """Release any model weights / GPU memory this backend holds.

        The base implementation is a no-op: hosted backends
        (:class:`LiteLLMBackend`) hold nothing on-device. Local backends
        (:class:`LocalTransformersBackend`) override it to free the model and
        empty the CUDA allocator, so a multi-phase pipeline can hold **one**
        model at a time instead of stacking several. After ``unload`` the backend
        can no longer generate."""
        return None


class LiteLLMBackend(LLMBackend):
    """Provider-agnostic backend routed through :mod:`gms.llm`.

    Accepts model names in the LiteLLM format ``<provider>/<id>`` (e.g.
    ``anthropic/claude-sonnet-4-20250514``, ``openai/gpt-4o``,
    ``ollama/llama3``). Internally constructs an :class:`gms.llm.LLMClient`
    with a custom model override and delegates ``.call()`` to its
    ``.complete()`` method.
    """

    def __init__(self, model: str):
        if not model or not model.strip():
            raise ValueError(
                "LiteLLMBackend requires a non-empty model name. "
                "Pass --model on the CLI, assign ConvertConfig.llm_model, "
                "or configure GMS_LLM_MODEL via gms.llm.LLMSettings."
            )
        from knowlytix.core.llm import LLMClient, LLMSettings

        self._model = model
        self._client = LLMClient(model=model, settings=LLMSettings())

    def call(self, system: str, user: str, max_tokens: int = 2048) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return self._client.complete(messages, max_tokens=max_tokens)

    @property
    def model_name(self) -> str:
        return self._model


class AnthropicBackend(LiteLLMBackend):
    """Backwards-compatible Anthropic-flavored backend.

    Accepts either a bare Claude model name (e.g.
    ``"claude-sonnet-4-20250514"``) or a LiteLLM-format name (e.g.
    ``"anthropic/claude-sonnet-4-20250514"``). Bare names are auto-
    prefixed with ``anthropic/`` before routing through LiteLLM so
    existing callers that encoded Anthropic-specific defaults keep
    working.

    Exposes the original caller-supplied model via ``bare_model_name``
    for code paths that still need to call the Anthropic SDK directly
    (e.g. provider-specific features like web_search_20250305 in
    ``knowlytix/knowledge/web_agent.py``).

    New code should instantiate :class:`LiteLLMBackend` directly with a
    provider-prefixed model name.
    """

    def __init__(self, model: str):
        if not model or not model.strip():
            raise ValueError(
                "AnthropicBackend requires a non-empty model name. "
                "Set ConvertConfig.llm_model, pass --model on the CLI, "
                "or configure GMS_LLM_MODEL via gms.llm.LLMSettings."
            )
        # Auto-prefix bare model names so callers passing
        # "claude-sonnet-4-20250514" still work. "foo/" or "/bar" are
        # malformed; treat as bare and re-prefix so LiteLLM receives a
        # valid "anthropic/<id>".
        trimmed = model.strip()
        head, sep, tail = trimmed.partition("/")
        if head and tail:
            resolved = trimmed
            bare = tail
        else:
            bare = trimmed.strip("/")
            resolved = f"anthropic/{bare}"
        self._bare_model = bare
        super().__init__(resolved)

    @property
    def bare_model_name(self) -> str:
        """Original caller-supplied model without any ``provider/`` prefix.

        Use this when invoking provider-specific SDK features that
        expect a bare model identifier.
        """

        return self._bare_model


class LocalTransformersBackend(LLMBackend):
    """Local HuggingFace model on GPU."""

    def __init__(self, model_name: str, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._model_name = model_name
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=device,
        )

    def call(self, system: str, user: str, max_tokens: int = 2048) -> str:
        """Greedily complete one prompt on the local model and return the decoded text.

        Args:
            system: optional system prompt prepended via the chat template.
            user: the user prompt.
            max_tokens: maximum number of new tokens to generate.

        Returns:
            The generated continuation with special tokens stripped.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        if hasattr(self._tokenizer, "apply_chat_template"):
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            text = f"{system}\n\n{user}" if system else user

        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
        outputs = self._model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=False, temperature=None, top_p=None,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def call_batch(self, users: "list[str]", system: str = "",
                   max_tokens: int = 2048) -> "list[str]":
        """Greedily complete many user prompts in ONE forward pass and return the
        texts in order. Same decoding as :meth:`call` (chat template, greedy), but
        the prompts are left-padded and decoded as a batch, so a rephrasing or
        evaluation loop over N messages costs one padded generate instead of N
        sequential calls. ``system`` is an optional prompt shared by every item.

        A single-item list is equivalent to :meth:`call`. This is the local,
        single-forward-pass override of the provider-agnostic batch entry point
        (``LLMClient.complete_batch``); callers batching a local model should use
        it rather than looping :meth:`call`."""
        import torch

        if not users:
            return []
        tok = self._tokenizer
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        prev_side = tok.padding_side
        tok.padding_side = "left"          # decoder-only batching needs left padding
        try:
            texts = []
            for user in users:
                msgs = ([{"role": "system", "content": system}] if system else []) \
                    + [{"role": "user", "content": user}]
                if hasattr(tok, "apply_chat_template"):
                    texts.append(tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True))
                else:
                    texts.append(f"{system}\n\n{user}" if system else user)
            enc = tok(texts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(self._device)
            with torch.no_grad():
                outputs = self._model.generate(
                    **enc, max_new_tokens=max_tokens, do_sample=False,
                    temperature=None, top_p=None, pad_token_id=tok.eos_token_id)
            gen = outputs[:, enc["input_ids"].shape[1]:]
            return [tok.decode(g, skip_special_tokens=True) for g in gen]
        finally:
            tok.padding_side = prev_side

    def unload(self) -> None:
        """Free the model and empty the CUDA allocator.

        Drops references to the model and tokenizer, runs a garbage collection,
        and calls ``torch.cuda.empty_cache()`` so the weights are reclaimed
        promptly rather than at interpreter GC. Use it between phases of a build
        that loads several local models in sequence (datagen → train → eval), so
        peak GPU memory is one model, not the sum. After ``unload`` this backend
        can no longer generate."""
        import gc

        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - torch always present on this path
            pass

    @property
    def model_name(self) -> str:
        return self._model_name


class LocalTransformersLLMClient:
    """A :class:`knowlytix.core.llm.LLMClient`-compatible client backed by a local
    HuggingFace model, so ``core.llm`` consumers — notably graphdoe's
    :class:`~knowlytix.harness.graphdoe.QuestionRephraser` — can run fully on-device.

    It implements the ``complete`` / ``complete_batch`` contract the rephraser
    depends on, and overrides ``complete_batch`` with the local single-forward-pass
    path (:meth:`LocalTransformersBackend.call_batch`), chunked so an
    ``expand``-sized cross product does not exceed GPU memory in one generate. This
    is the local-engine override the ``LLMClient.complete_batch`` docstring invites.
    """

    def __init__(self, model_name: str, *, device: str = "cuda",
                 backend: "LocalTransformersBackend | None" = None,
                 batch_chunk: int = 32, max_tokens_cap: int | None = None):
        self.model = model_name
        self.settings = None
        self._be = backend or LocalTransformersBackend(model_name, device=device)
        self._chunk = batch_chunk
        # Cap generation length regardless of the caller's request. Consumers like
        # QuestionRephraser ask for a generous ceiling (1024) but produce a short
        # rewrite; capping avoids runaway greedy decoding to that ceiling on a local
        # model. None = respect the caller's max_tokens.
        self._cap = max_tokens_cap

    def _mt(self, kwargs) -> int:
        mt = int(kwargs.get("max_tokens", 512))
        return min(mt, self._cap) if self._cap else mt

    @staticmethod
    def _split(messages) -> "tuple[str, str]":
        msgs = list(messages)
        system = " ".join(m["content"] for m in msgs if m.get("role") == "system")
        user = "\n\n".join(m["content"] for m in msgs if m.get("role") != "system")
        return system, user

    def complete(self, messages, **kwargs) -> str:
        system, user = self._split(messages)
        return self._be.call(system, user, max_tokens=self._mt(kwargs))

    def complete_batch(self, messages_list, **kwargs) -> "list[str]":
        pairs = [self._split(m) for m in messages_list]
        if not pairs:
            return []
        max_tokens = self._mt(kwargs)
        # The rephraser shares one system prompt across the batch; if that holds we
        # can use the single-pass call_batch, chunked to bound GPU memory. A mixed
        # batch falls back to per-item complete.
        systems = {s for s, _ in pairs}
        if len(systems) > 1:
            return [self._be.call(s, u, max_tokens=max_tokens) for s, u in pairs]
        system = next(iter(systems)) if systems else ""
        users = [u for _, u in pairs]
        out: list[str] = []
        for i in range(0, len(users), self._chunk):
            out.extend(self._be.call_batch(users[i:i + self._chunk], system=system,
                                           max_tokens=max_tokens))
        return out


def create_backend(config: ConvertConfig) -> LLMBackend:
    """Factory: create LLM backend from config.

    Returns :class:`LocalTransformersBackend` when the user requested the
    local backend and supplied a ``local_model_name``; otherwise
    constructs an :class:`AnthropicBackend` (which delegates to
    :class:`LiteLLMBackend` via LiteLLM).

    Raises :class:`ValueError` if no model name has been supplied —
    surfaces the misconfiguration before the first (failing) API call.
    """

    if config.llm_backend == "local" and config.local_model_name:
        return LocalTransformersBackend(config.local_model_name)
    if not config.llm_model:
        raise ValueError(
            "knowlytix.knowledge: ConvertConfig.llm_model is empty. Either pass "
            "--model on the CLI, assign `config.convert.llm_model` "
            "directly, or configure GMS_LLM_MODEL via "
            "gms.llm.LLMSettings."
        )
    return AnthropicBackend(config.llm_model)
