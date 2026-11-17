# SPDX-License-Identifier: Apache-2.0
"""EVD005: an optional model review of whether the cited evidence entails each fact.

The other evidence rules check citations lexically. This one asks a language
model, supplied by the caller, whether the cited passages state the fact,
contradict it, or do not settle it, attending to negation, quantities (value,
unit and magnitude) and qualifiers (a fact stated only under a condition).
The model must quote the passage text its verdict rests on; a supported or
contradicted verdict whose quote is not in the passages becomes an abstention,
and so does an answer that does not parse. Verdicts are advisory: the model,
the prompt version and every quote are recorded, and nothing is calibrated
unless ``kgcheck.evaluation`` has measured it on held-out labeled data.

The rule is not registered with the shipped rules, because it needs a model.
Build it with ``semantic_review_rule(SemanticReviewer(backend, model=...))`` and
pass it to ``audit(..., rules=[...])``. ``backend`` is any object with
``call(system, user, max_tokens) -> str``, the protocol of OntoDisc's
extraction backends; kgcheck does not choose or load a model itself.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from knowlytix.knowledge.kgcheck.models import EvidenceLink, GraphView, Severity
from knowlytix.knowledge.kgcheck.registry import Outcome, Rule, RuleContext, RuleSpec
from knowlytix.knowledge.kgcheck.rules._common import rows_by_assertion, triple
from knowlytix.knowledge.kgcheck.sources import split_sections

PROMPT_VERSION = "kgcheck-review/1"
LABELS = ("supported", "contradicted", "not_enough_info")
ABSTAIN = "abstain"
_SECTION_LEVEL = frozenset({"structure", "source_span"})

SYSTEM = """You check one knowledge-graph fact against the source passages cited for it.
The fact is a triple [subject, relation, object]. Decide which holds:
- "supported": the passages state the fact for this subject, with the same value, unit and
  magnitude, without a negation, and not only under a condition the fact omits;
- "contradicted": the passages state something incompatible for this subject: another value
  for the same quantity, the negation of the fact, or a different subject for the value;
- "not_enough_info": the passages neither state nor contradict it, or state it only under a
  condition or for a different scope than the fact.
Quote, exactly as written in the passages, the shortest passage text your answer rests on
(for not_enough_info, the closest passage text, or an empty string).
Answer with one JSON object and nothing else:
{"label": "supported" | "contradicted" | "not_enough_info", "quote": "...", "reason": "..."}"""


class Backend(Protocol):
    def call(self, system: str, user: str, max_tokens: int) -> str: ...


@dataclass(frozen=True)
class Verdict:
    """One review. ``label`` is one of ``LABELS`` or ``abstain``; ``why`` says why it abstained."""

    label: str
    quote: str = ""
    reason: str = ""
    why: str | None = None


_QUOTES = str.maketrans({"\u201c": "'", "\u201d": "'", "\u2018": "'", "\u2019": "'", '"': "'"})


def _normal(text: str) -> str:
    """Whitespace collapsed and quotation marks unified, for verbatim-quote matching."""
    return re.sub(r"\s+", " ", text.translate(_QUOTES)).strip().lower()


class SemanticReviewer:
    """Asks ``backend`` for a verdict on one fact and validates the answer."""

    def __init__(self, backend: Backend, *, model: str, max_tokens: int = 512) -> None:
        self.backend, self.model, self.max_tokens = backend, model, max_tokens

    def review(self, fact: Sequence[str], passages: Sequence[str]) -> Verdict:
        user = json.dumps({"fact": list(fact), "passages": list(passages)}, ensure_ascii=False)
        return parse_verdict(self.backend.call(SYSTEM, user, self.max_tokens), passages)


MIN_QUOTE = 8  # characters, after normalization; shorter quotes ("5") prove nothing


def quoted(quote: str, passages: Sequence[str]) -> bool:
    """True when ``quote`` occurs in one passage, starting and ending on word boundaries.

    A quote may not span two passages, and may not end inside a token: "limit of $5"
    is not a quote of "limit of $50,000". Quotes shorter than ``MIN_QUOTE`` count
    only when they are a whole passage.
    """
    q = _normal(quote)
    if not q:
        return False
    for passage in passages:
        text = _normal(passage)
        if len(q) < MIN_QUOTE and q != text:
            continue
        pattern = r"(?<![\w$€£¥.,])" + re.escape(q) + r"(?![\w])(?![.,]\d)"
        if re.search(pattern, text):
            return True
    return False


def parse_verdict(raw: str, passages: Sequence[str]) -> Verdict:
    """The verdict in a model answer, or an abstention saying what was wrong with it.

    A not_enough_info verdict needs no quote, but a quote it gives that is not in the
    passages is dropped rather than recorded."""
    answer = _json_object(raw)
    if answer is None:
        return Verdict(ABSTAIN, why="unparsable")
    label, quote = answer.get("label"), answer.get("quote", "")
    reason = answer.get("reason", "")
    if label not in LABELS or not isinstance(quote, str):
        return Verdict(ABSTAIN, why="invalid_label")
    reason = reason if isinstance(reason, str) else ""
    if quoted(quote, passages):
        return Verdict(label, quote, reason)
    if label == "not_enough_info":
        return Verdict(label, "", reason)
    return Verdict(ABSTAIN, quote, reason, why="unquoted")


def _json_object(raw: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def passages_for(links: Sequence[EvidenceLink], sections: dict[str, str], max_chars: int) -> list[str]:
    """The cited text of each link; a section-level citation contributes its section, cut."""
    out: list[str] = []
    for link in links:
        text = sections.get(link.section, link.sentence) if link.kind in _SECTION_LEVEL \
            else (link.span or link.sentence)
        text = text[:max_chars]
        if text and text not in out:
            out.append(text)
    return out


def semantic_review_rule(reviewer: SemanticReviewer, *, max_facts: int | None = None,
                         max_chars: int = 4000) -> Rule:
    """EVD005 with this reviewer. ``max_facts`` reviews only the first facts (partial)."""
    spec = RuleSpec(
        "EVD005", "1", "evidence", "Model review of evidence entailment", Severity.WARNING,
        phase="E", prerequisites=("provenance",), views=(GraphView.STORED, GraphView.CONFORMANT),
        cacheable=False,
        basis=f"a language model ({reviewer.model}, prompt {PROMPT_VERSION}) judged each fact "
              "against its cited passages; a verdict without a verbatim quote from them is an "
              "abstention. Advisory and uncalibrated unless measured on held-out labeled data")

    def check(ctx: RuleContext) -> Outcome:
        return _review(ctx, reviewer, max_facts, max_chars)
    return Rule(spec, check)


def _review(ctx: RuleContext, reviewer: SemanticReviewer, max_facts: int | None,
            max_chars: int) -> Outcome:
    snap = ctx.snapshot
    sections = split_sections(snap.sources[0].text) if snap.sources else {}
    links: dict[str, list[EvidenceLink]] = {}
    for link in snap.evidence:
        links.setdefault(link.assertion_id, []).append(link)
    targets = sorted(a for a in rows_by_assertion(snap, ctx.view) if a in links)
    chosen = targets if max_facts is None else targets[:max_facts]
    counts = {label: 0 for label in (*LABELS, ABSTAIN)}
    reasons: dict[str, int] = {}
    findings, failed = [], 0
    for aid in chosen:
        fact = triple(snap, aid)
        passages = passages_for(links[aid], sections, max_chars)
        try:
            verdict = reviewer.review(fact, passages)
        except Exception:  # noqa: BLE001  # a backend failure leaves this fact unreviewed
            failed += 1
            continue
        counts[verdict.label] += 1
        if verdict.why:
            reasons[verdict.why] = reasons.get(verdict.why, 0) + 1
        if verdict.label in ("contradicted", "not_enough_info"):
            findings.append(ctx.finding(
                f"the reviewer reads the evidence for {list(fact)} as "
                f"{verdict.label.replace('_', ' ')}",
                severity=Severity.WARNING if verdict.label == "contradicted" else Severity.INFO,
                affected=(aid,), locations=tuple(lk.raw for lk in links[aid][:5]),
                discriminator=verdict.label,
                witness={"fact": list(fact), "label": verdict.label, "quote": verdict.quote[:500],
                         "reason": verdict.reason[:500], "model": reviewer.model,
                         "prompt_version": PROMPT_VERSION},
                suggestion="review the fact against the quoted passage"))
    abstained = counts[ABSTAIN]
    notes = [f"reviewed the first {len(chosen)} of {len(targets)} facts (max_facts)"
             if len(chosen) < len(targets) else "", f"the backend failed on {failed} facts"
             if failed else "", f"the reviewer abstained on {abstained} facts" if abstained else ""]
    # an abstention is not a review: only committed verdicts count as tested
    return Outcome(eligible=len(targets), tested=len(chosen) - failed - abstained,
                   findings=tuple(findings),
                   partial="; ".join(n for n in notes if n) or None,
                   metrics={"model": reviewer.model, "prompt_version": PROMPT_VERSION,
                            "verdicts": counts, "abstention_reasons": dict(sorted(reasons.items())),
                            "backend_failures": failed,
                            "advisory": True})
