# SPDX-License-Identifier: Apache-2.0
"""LLM-generated aliases — supervision for the GEODE embedding loop.

A document rarely declares its own synonyms. The annual report names a segment
``cloud platform`` and never records that a reader might call it "the cloud
unit", that ``devices`` is "consumer hardware", or that the ``has_revenue``
property is the "topline". Without those surface forms the GEODE embedding loop
(:mod:`knowlytix.knowledge.geode.embed_loop`) has no alias structure to learn
from, so it cannot tune the encoder the binder and the query parser rely on.

This module fills that gap *without hand-authored aliases*. It asks a local LLM
(the wireable GEODE actor, default Qwen 4B — see
:mod:`knowlytix.knowledge.geode.agent_llm`) to propose plausible surface forms,
then validates and de-collides them. Two kinds are generated:

* :func:`generate_entity_aliases` — surface forms for each canonical *entity*
  ("consumer hardware" -> ``devices``), which improve head binding;
* :func:`generate_relation_aliases` — natural-language phrasings for each
  *relation* ("how many people" -> ``has_headcount``), which improve relation
  discrimination (the parser scores a question against each relation's phrase, so
  it must tell ``has_headcount`` from ``has_revenue``).

The result is a ``{canonical -> [aliases]}`` map used **only as encoder-SFT
supervision**. It is never inserted into the graph: the stored triples and ENM
stay built from the original corpus alone, while the encoder learns the wider
vocabulary. The aliases are a build-time artifact, so callers persist them next
to the store (``generated_aliases.json`` / ``generated_relation_aliases.json``)
for reproducibility and audit.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Iterable

from knowlytix.knowledge.geode.agent_llm import DEFAULT_AGENT_MODEL, agent_callable

__all__ = [
    "generate_entity_aliases",
    "generate_relation_aliases",
    "ALIAS_SYSTEM_PROMPT",
    "RELATION_SYSTEM_PROMPT",
]


ALIAS_SYSTEM_PROMPT = (
    "You generate alternative ways a user might refer to a named entity in a "
    "document. Given one canonical entity name (and the other entities in the "
    "same document), list short surface forms a real user could type to mean "
    "that SAME entity: nicknames, abbreviations, paraphrases and common "
    "synonyms. Rules: (1) every alias must mean the given entity and NONE of "
    "the other listed entities; (2) do not invent facts or numbers; (3) keep "
    "each alias to at most a few words; (4) do not repeat the canonical name "
    "itself. Respond with ONLY a JSON array of strings, e.g. "
    '["the cloud unit", "cloud services"].'
)

RELATION_SYSTEM_PROMPT = (
    "You generate alternative ways a user might ask about a property in a "
    "document. Given one canonical property (and the other properties in the "
    "same document), list short natural-language phrasings a user could type to "
    "ask for that SAME property: question wordings, synonyms and common "
    "phrasings. Rules: (1) every phrasing must mean the given property and NONE "
    "of the other listed properties; (2) do not invent facts or values; "
    "(3) keep each phrasing short; (4) do not repeat the canonical name itself. "
    "Respond with ONLY a JSON array of strings, e.g. "
    '["how many people", "number of employees", "staff count"].'
)


def _humanize(token: str) -> str:
    """``cloud_platform`` -> ``cloud platform`` — a natural surface form."""
    return token.replace("_", " ").replace("/", " ").strip()


def _relation_phrase(rel: str) -> str:
    """``has_headcount`` -> ``headcount`` — matches the query parser's phrasing."""
    base = rel[4:] if rel.startswith("has_") else rel
    return base.replace("_", " ").strip()


def _norm(text: str) -> str:
    """Lowercase, collapse whitespace — the canonical form for comparison."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _parse_alias_list(raw: str) -> list[str]:
    """Pull a list of alias strings out of an LLM response.

    Prefers the first JSON array in the text; falls back to splitting lines /
    commas when the model does not emit clean JSON.
    """
    m = re.search(r"\[.*?\]", raw, flags=re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return [str(x) for x in arr]
        except (json.JSONDecodeError, ValueError):
            pass
    parts = re.split(r"[\n,]+", raw)
    out = []
    for p in parts:
        p = p.strip().strip("-*•").strip().strip('"').strip("'").strip()
        if p and not p.lower().startswith(("here", "alias", "json", "[", "]")):
            out.append(p)
    return out


def _build_prompt(name: str, siblings: list[str], context: str | None,
                  n_aliases: int, *, system_prompt: str,
                  humanizer: Callable[[str], str], noun: str) -> str:
    """Assemble the per-name user prompt (system rules prepended)."""
    human = humanizer(name)
    others = ", ".join(humanizer(s) for s in siblings if s != name)
    lines = [system_prompt, "", f'Canonical {noun}: "{human}"']
    if others:
        lines.append(f"Other {noun} values in the document (do NOT alias these): "
                     f"{others}")
    if context:
        lines.append("")
        lines.append("Document context (for grounding, do not quote numbers):")
        lines.append(context.strip()[:1500])
    lines.append("")
    lines.append(f'List up to {n_aliases} aliases for "{human}" as a JSON array:')
    return "\n".join(lines)


def _generate(names: Iterable[str], llm, *, system_prompt: str,
              humanizer: Callable[[str], str], noun: str,
              context: "str | dict[str, str] | None", n_aliases: int,
              block_values: "Iterable[str] | None", model_name: str, device: str,
              max_tokens: int, allow_frontier: bool, drop_collisions: bool,
              max_alias_words: int) -> dict[str, list[str]]:
    """Core generator shared by the entity and relation entry points."""
    items = list(dict.fromkeys(names))
    actor = agent_callable(llm, model_name=model_name, device=device,
                           max_tokens=max_tokens, allow_frontier=allow_frontier)

    # A canonical's own humanized/raw form is a binding target on its own; any
    # known value (a fact, not a name) is barred outright.
    reserved = {_norm(humanizer(n)) for n in items} | {_norm(n) for n in items}
    blocked = {_norm(v) for v in (block_values or [])} | {
        _norm(humanizer(v)) for v in (block_values or [])}

    raw: dict[str, list[str]] = {}
    for name in items:
        ctx = context.get(name) if isinstance(context, dict) else context
        prompt = _build_prompt(name, items, ctx, n_aliases,
                               system_prompt=system_prompt, humanizer=humanizer,
                               noun=noun)
        try:
            resp = actor(prompt)
        except Exception:  # noqa: BLE001 — one item's failure is not fatal
            raw[name] = []
            continue
        self_forms = {_norm(name), _norm(humanizer(name))}
        seen: set[str] = set()
        kept: list[str] = []
        for c in _parse_alias_list(resp):
            a = _norm(c)
            if not a or a in seen or a in self_forms:
                continue
            if len(a.split()) > max_alias_words:
                continue
            if not re.search(r"[a-z]", a):
                continue
            if a in blocked:
                continue
            if a in reserved and a not in self_forms:
                continue
            seen.add(a)
            kept.append(a)
        raw[name] = kept[:n_aliases]

    if drop_collisions:
        counts: dict[str, int] = {}
        for kept in raw.values():
            for a in kept:
                counts[a] = counts.get(a, 0) + 1
        raw = {n: [a for a in kept if counts[a] == 1] for n, kept in raw.items()}
    return raw


def generate_entity_aliases(
    entities: Iterable[str],
    llm: "Callable[[str], str] | object | str | None" = None,
    *,
    context: "str | dict[str, str] | None" = None,
    n_aliases: int = 6,
    block_values: "Iterable[str] | None" = None,
    model_name: str = DEFAULT_AGENT_MODEL,
    device: str = "cuda",
    max_tokens: int = 192,
    allow_frontier: bool = False,
    drop_collisions: bool = True,
    max_alias_words: int = 6,
) -> dict[str, list[str]]:
    """Generate ``{canonical entity -> [aliases]}`` with a local LLM.

    Args:
        entities: the canonical entity names (e.g. the graph's heads).
        llm: the actor LLM, wired via
            :func:`~knowlytix.knowledge.geode.agent_llm.agent_callable` (a
            callable, an ``LLMBackend``, a model-name string, or ``None`` to
            build a local backend for ``model_name``, default Qwen 4B).
        context: optional grounding text — a shared string or a
            ``{entity -> text}`` dict.
        n_aliases: max aliases requested per entity.
        block_values: surface forms that must never become an alias — pass the
            graph's tail values so a fact value ("morgan lee") cannot leak in.
        drop_collisions: drop an alias proposed for more than one entity (an
            ambiguous surface that would teach a confident wrong binding).
        max_alias_words: drop aliases longer than this many words.

    Returns:
        ``{entity -> [aliases]}`` for every input entity (possibly empty),
        normalized, de-duplicated and collision-filtered. Supervision only —
        never written into the graph.
    """
    return _generate(
        entities, llm, system_prompt=ALIAS_SYSTEM_PROMPT, humanizer=_humanize,
        noun="entity", context=context, n_aliases=n_aliases,
        block_values=block_values, model_name=model_name, device=device,
        max_tokens=max_tokens, allow_frontier=allow_frontier,
        drop_collisions=drop_collisions, max_alias_words=max_alias_words)


def generate_relation_aliases(
    relations: Iterable[str],
    llm: "Callable[[str], str] | object | str | None" = None,
    *,
    context: "str | dict[str, str] | None" = None,
    n_aliases: int = 6,
    model_name: str = DEFAULT_AGENT_MODEL,
    device: str = "cuda",
    max_tokens: int = 192,
    allow_frontier: bool = False,
    drop_collisions: bool = True,
    max_alias_words: int = 6,
) -> dict[str, list[str]]:
    """Generate ``{canonical relation -> [phrasings]}`` with a local LLM.

    The relation is humanized the way the query parser phrases it —
    ``has_headcount`` -> ``headcount`` — so the generated phrasings ("how many
    people", "number of employees") cluster with the phrase the parser actually
    encodes at query time. This is what lets the tuned encoder tell
    ``has_headcount`` from ``has_revenue``, the relation-discrimination gap a
    purely entity-aliased encoder leaves open.

    Args mirror :func:`generate_entity_aliases` (without ``block_values`` — a
    relation has no tail value to bar). Returns ``{relation -> [phrasings]}``,
    keyed by the original relation name; supervision only, never written to the
    graph.
    """
    return _generate(
        relations, llm, system_prompt=RELATION_SYSTEM_PROMPT,
        humanizer=_relation_phrase, noun="property", context=context,
        n_aliases=n_aliases, block_values=None, model_name=model_name,
        device=device, max_tokens=max_tokens, allow_frontier=allow_frontier,
        drop_collisions=drop_collisions, max_alias_words=max_alias_words)
