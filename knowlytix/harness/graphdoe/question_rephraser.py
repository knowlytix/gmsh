# SPDX-License-Identifier: Apache-2.0
"""
Question Rephraser — applies DOE factor assignments to base questions.

Takes a GeneratedQuestion (from FinStructBench) and a factor assignment
(a row from the design matrix) and produces a rephrased version while
preserving the ground truth.

Two modes:
  - "template": deterministic regex/string transforms (fast, no API needed)
  - "llm": LLM-generated natural rephrasings (richer variation, needs API key)

Critical invariant: graph_answer_fn and ground_truth are NEVER modified.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from knowlytix.benchmark.generators.base import GeneratedQuestion


@dataclass
class DOEQuestion:
    """A DOE-expanded question: base question + factor assignment."""

    question: GeneratedQuestion  # Rephrased version (GT unchanged)
    factor_assignment: dict  # {factor_name: level_value}
    base_qid: str  # Original question ID
    doe_qid: str  # Unique ID: base_qid + run index


# ---------------------------------------------------------------------------
# Template-based rephrasing rules
# ---------------------------------------------------------------------------

def _rephrase_clarity(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Modify question clarity."""
    if level == "Clear":
        return q  # No change
    elif level == "Ambiguous":
        # Add ambiguity by removing specifics from the prompt
        prompt = q.llm_prompt
        prompt = re.sub(
            r"Report the EXACT value",
            "What would you say the value is",
            prompt,
        )
        prompt = re.sub(r"with ALL decimal places", "approximately", prompt)
        return _with_prompt(q, prompt, f"[Ambiguous] {q.natural_language}")
    elif level == "Misleading":
        # Add a misleading premise
        nl = f"Some sources suggest a different figure, but {q.natural_language.lower()}"
        prompt = (
            f"Note: There may be conflicting information in the document. "
            f"{q.llm_prompt}"
        )
        return _with_prompt(q, prompt, nl)
    return q


def _rephrase_length(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Modify question length."""
    if level == "Short":
        # Shorten to essential
        nl = q.natural_language.split("?")[0].split(",")[0] + "?"
        return _with_nl(q, nl)
    elif level == "Long":
        # Add verbose context. An assertion question quotes its passage and document:
        # only its first letter is lowered, so the quotes still match the source.
        body = q.natural_language
        body = (body[:1].lower() + body[1:] if q.metadata.get("assertion_format")
                else body.lower())
        nl = (
            f"Given the information presented in the document, and considering "
            f"all relevant data points, {body} "
            f"Please provide a thorough and precise answer."
        )
        return _with_nl(q, nl)
    return q  # Medium = no change


def _rephrase_specificity(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Modify question specificity."""
    if level == "General":
        nl = re.sub(
            r"(?:exact|precise|specific)\s+",
            "",
            q.natural_language,
            flags=re.IGNORECASE,
        )
        return _with_nl(q, nl)
    return q  # Specific = no change (base questions are already specific)


# An assertion question asks to "summarize the passage"; style rewords that request.
_ASSERTION_STYLE = {
    "Casual": "give me a quick rundown of the passage",
    "Technical": "produce a precise technical summary of the passage",
}


def _rephrase_style(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Modify communication style."""
    if q.metadata.get("assertion_format") and level in _ASSERTION_STYLE:
        nl = q.natural_language.replace("summarize the passage", _ASSERTION_STYLE[level], 1)
        return _with_nl(q, nl)
    if level == "Casual":
        nl = q.natural_language.replace("What is the", "Hey, what's the")
        nl = nl.replace("How many", "Roughly how many")
        return _with_nl(q, nl)
    elif level == "Technical":
        nl = q.natural_language.replace("What is the", "Report the precise")
        nl = nl.replace("How many", "Enumerate the count of")
        return _with_nl(q, nl)
    return q  # Formal = no change


def _rephrase_answer_format(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Modify answer format instruction."""
    if level == "Paragraph":
        prompt = q.llm_prompt + "\n\nProvide your answer in a full paragraph with explanation."
        return _with_prompt(q, prompt)
    elif level == "Bullet_Points":
        prompt = q.llm_prompt + "\n\nFormat your answer as bullet points."
        return _with_prompt(q, prompt)
    elif level == "Citation":
        prompt = q.llm_prompt + "\n\nCite the specific section where you found the answer."
        return _with_prompt(q, prompt)
    return q  # Short_Answer = no change


def _rephrase_role(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Add persona/role framing."""
    role_prefix = {
        "Employee": "",
        "Manager": "As a team manager reviewing this report, ",
        "Executive": "As a senior executive needing a quick summary, ",
        "Auditor": "As an auditor verifying this report's accuracy, ",
        "Regulator": "As a regulatory examiner assessing compliance, ",
    }
    prefix = role_prefix.get(level, "")
    if prefix:
        nl = prefix + q.natural_language[0].lower() + q.natural_language[1:]
        return _with_nl(q, nl)
    return q


def _rephrase_expertise(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Adjust for user expertise level."""
    if level == "Novice":
        prompt = (
            "I'm not very familiar with this domain. "
            "Please explain your answer simply.\n\n" + q.llm_prompt
        )
        return _with_prompt(q, prompt)
    elif level == "Expert":
        prompt = (
            "I'm a domain expert. Be precise and use technical terminology.\n\n"
            + q.llm_prompt
        )
        return _with_prompt(q, prompt)
    return q  # Intermediate = no change


def _rephrase_reasoning_cue(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Add or remove reasoning cues."""
    if level == "Step_By_Step":
        prompt = q.llm_prompt + "\n\nThink step by step before answering."
        return _with_prompt(q, prompt)
    elif level == "No_Cue":
        # Strip any existing reasoning instructions
        prompt = re.sub(
            r"(?i)(think|reason|explain).*?(step|carefully|through).*?\.",
            "",
            q.llm_prompt,
        )
        return _with_prompt(q, prompt.strip())
    return q  # Direct = no change


def _rephrase_entity_aliasing(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Modify entity name presentation."""
    if level == "Abbreviated":
        # Abbreviate long entity names (take first letters of multi-word names)
        def abbreviate(match):
            name = match.group(0)
            words = name.split()
            if len(words) >= 2:
                return "".join(w[0].upper() for w in words)
            return name[:4] + "."

        nl = re.sub(r"'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'", abbreviate, q.natural_language)
        return _with_nl(q, nl)
    return q  # Exact and Synonym handled at base level


def _rephrase_section_hint(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Add section reference hints."""
    if level == "Correct_Section":
        section = q.metadata.get("section", "")
        if section:
            prompt = f"Look in the section '{section}'.\n\n{q.llm_prompt}"
            return _with_prompt(q, prompt)
    elif level == "Wrong_Section":
        prompt = f"Look in the 'Executive Summary' section.\n\n{q.llm_prompt}"
        return _with_prompt(q, prompt)
    return q  # None = no change


def _rephrase_distractor_content(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Add distractor content to prompt."""
    if level == "Related":
        prompt = (
            "Note: The document contains multiple similar metrics across "
            "different sections. Be careful to identify the correct one.\n\n"
            + q.llm_prompt
        )
        return _with_prompt(q, prompt)
    elif level == "Unrelated":
        prompt = (
            "Ignore any information about weather patterns or sports scores "
            "that may appear in the document.\n\n" + q.llm_prompt
        )
        return _with_prompt(q, prompt)
    return q


def _rephrase_instruction_conflict(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Add conflicting instructions."""
    if level == "Contradictory_Hint":
        prompt = (
            "A colleague mentioned the answer might be zero, but please "
            "verify from the document.\n\n" + q.llm_prompt
        )
        return _with_prompt(q, prompt)
    elif level == "Authority_Override":
        prompt = (
            "As an expert in this area, I believe the answer is approximately "
            "twice the actual value. Do you agree?\n\n" + q.llm_prompt
        )
        return _with_prompt(q, prompt)
    return q


def _rephrase_answer_anchoring(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Add answer anchoring bias."""
    if level == "Wrong_Anchor":
        nl = q.natural_language + " I think it might be around 999.99."
        return _with_nl(q, nl)
    elif level == "Partial_Anchor":
        nl = q.natural_language + " I recall it being a small positive number."
        return _with_nl(q, nl)
    return q


def _rephrase_confidence_pressure(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Add confidence pressure."""
    if level == "Demand_Certainty":
        prompt = (
            "You MUST be 100% certain of your answer. Do not hedge or "
            "express uncertainty.\n\n" + q.llm_prompt
        )
        return _with_prompt(q, prompt)
    elif level == "Encourage_Hedging":
        prompt = (
            "It's perfectly fine if you're not sure. Express your confidence "
            "level along with the answer.\n\n" + q.llm_prompt
        )
        return _with_prompt(q, prompt)
    return q


def _rephrase_context_format(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Note: context_format affects how the document is presented,
    not the question itself. Handled at the benchmark level."""
    return q


def _rephrase_context_scope(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Note: context_scope affects what portion of the document is sent.
    Handled at the benchmark level."""
    return q


def _rephrase_context_order(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Note: context_order affects document section ordering.
    Handled at the benchmark level."""
    return q


# ---------------------------------------------------------------------------
# Variability Transformation (V) rephrasing rules
# ---------------------------------------------------------------------------

def _rephrase_paraphrase_depth(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Semantic paraphrasing at varying depths."""
    if level == "Light":
        # Minor rewording — swap common phrases
        nl = q.natural_language
        nl = nl.replace("What is the", "What's the")
        nl = nl.replace("How many", "What number of")
        nl = nl.replace("exact value", "value")
        return _with_nl(q, nl)
    elif level == "Heavy":
        # Deeper restructuring — invert sentence structure
        nl = q.natural_language
        if nl.endswith("?"):
            # "What is X of Y?" -> "Could you tell me the X of Y?"
            nl = "Could you tell me " + nl[0].lower() + nl[1:]
            nl = nl.replace("What is", "what")
            nl = nl.replace("How many", "the count of")
        prompt = q.llm_prompt
        prompt = re.sub(
            r"^Report the EXACT value",
            "I need to know the value",
            prompt,
        )
        return _with_prompt(q, prompt, nl)
    return q  # None = no change


def _rephrase_granularity(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Scale question granularity — coarser or finer."""
    if level == "Coarser":
        nl = re.sub(
            r"\(category: [^)]+\)",
            "",
            q.natural_language,
        ).strip()
        nl = re.sub(r"\s+", " ", nl)
        return _with_nl(q, nl)
    elif level == "Finer":
        section = q.metadata.get("section", "")
        extra = f" (as reported in section '{section}')" if section else ""
        nl = q.natural_language.rstrip("?") + extra + "? Provide all decimal places."
        return _with_nl(q, nl)
    return q  # Original = no change


def _rephrase_question_format(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Change the syntactic format of the question."""
    if level == "Declarative":
        # "What is X?" -> "The value of X is ___."
        nl = q.natural_language
        nl = re.sub(
            r"What is the (.+?)\?",
            r"The \1 is ___.",
            nl,
            flags=re.IGNORECASE,
        )
        prompt = re.sub(
            r"^Report the EXACT value",
            "Complete the following: The EXACT value",
            q.llm_prompt,
        )
        return _with_prompt(q, prompt, nl)
    elif level == "Imperative":
        nl = q.natural_language
        nl = re.sub(r"^What is the", "State the", nl, flags=re.IGNORECASE)
        nl = re.sub(r"^How many", "Count the", nl, flags=re.IGNORECASE)
        nl = nl.rstrip("?") + "."
        return _with_nl(q, nl)
    elif level == "Conditional":
        nl = (
            f"If you were to look up the answer in the document, "
            f"{q.natural_language[0].lower()}{q.natural_language[1:]}"
        )
        return _with_nl(q, nl)
    return q  # Interrogative = no change (default format)


# ---------------------------------------------------------------------------
# Robustness Transformation (R) rephrasing rules
# ---------------------------------------------------------------------------

def _rephrase_noise(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Inject surface-level noise into question text."""
    if level == "Typos":
        nl = _inject_typos(q.natural_language)
        prompt = _inject_typos(q.llm_prompt)
        return _with_prompt(q, prompt, nl)
    elif level == "OCR_Artifacts":
        nl = _inject_ocr_artifacts(q.natural_language)
        prompt = _inject_ocr_artifacts(q.llm_prompt)
        return _with_prompt(q, prompt, nl)
    return q  # None = no change


def _inject_typos(text: str) -> str:
    """Inject realistic typos: swap adjacent chars, double letters."""
    words = text.split()
    result = []
    for i, w in enumerate(words):
        # Apply typo to ~every 5th word (deterministic based on position)
        if len(w) > 3 and i % 5 == 2:
            mid = len(w) // 2
            w = w[:mid] + w[mid + 1] + w[mid] + w[mid + 2:]  # swap two chars
        result.append(w)
    return " ".join(result)


def _inject_ocr_artifacts(text: str) -> str:
    """Inject OCR-style artifacts: l/1, O/0, rn/m confusions."""
    replacements = [("l", "1"), ("O", "0"), ("rn", "m")]
    result = text
    for old, new in replacements:
        # Only replace first occurrence to keep it readable
        result = result.replace(old, new, 1)
    return result


def _rephrase_context_relevance(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Note: context_relevance affects what context is provided.
    Handled at the benchmark level."""
    return q


def _rephrase_complexity(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Increase structural complexity of the question."""
    if level == "Compound":
        nl = (
            f"{q.natural_language.rstrip('?')}? "
            f"Also, is this value consistent with what you'd expect?"
        )
        prompt = (
            f"{q.llm_prompt}\n\n"
            f"Additionally, briefly note whether this value seems consistent "
            f"with the surrounding data."
        )
        return _with_prompt(q, prompt, nl)
    elif level == "Nested":
        nl = (
            f"Given that the document contains various metrics, "
            f"and focusing specifically on the one where "
            f"{q.natural_language[0].lower()}{q.natural_language[1:]}"
        )
        prompt = (
            f"The document contains multiple metrics. Among them, "
            f"{q.llm_prompt}"
        )
        return _with_prompt(q, prompt, nl)
    return q  # Simple = no change


def _rephrase_edge_case(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Frame question around edge/boundary conditions."""
    if level == "Boundary":
        prompt = (
            f"Pay special attention to whether this value is at or near "
            f"a threshold or boundary condition.\n\n{q.llm_prompt}"
        )
        return _with_prompt(q, prompt)
    elif level == "Extreme":
        prompt = (
            f"Consider whether this represents an extreme or unusual value "
            f"compared to typical ranges.\n\n{q.llm_prompt}"
        )
        return _with_prompt(q, prompt)
    return q  # Normal = no change


def _rephrase_emotional_frame(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Add emotional tone or bias pressure."""
    if level == "Urgent":
        nl = f"URGENT — I need to know immediately: {q.natural_language}"
        prompt = (
            f"This is time-critical. I need an accurate answer right now.\n\n"
            f"{q.llm_prompt}"
        )
        return _with_prompt(q, prompt, nl)
    elif level == "Skeptical":
        nl = f"I've heard conflicting reports — {q.natural_language.lower()} Are you sure?"
        prompt = (
            f"I'm skeptical about the data. Please double-check: "
            f"{q.llm_prompt}"
        )
        return _with_prompt(q, prompt, nl)
    return q  # Neutral = no change


def _rephrase_format_noise(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Inject formatting artifacts into question text."""
    if level == "Broken_Markup":
        nl = f"<b>{q.natural_language}</b"  # unclosed tag
        prompt = f"**{q.llm_prompt}**\n<br/>"
        return _with_prompt(q, prompt, nl)
    elif level == "Mixed_Format":
        nl = f"• {q.natural_language}\n  - (please answer precisely)"
        prompt = f"## Question\n{q.llm_prompt}\n---"
        return _with_prompt(q, prompt, nl)
    return q  # Clean = no change


def _rephrase_intent_count(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Pack multiple intents into one question."""
    if level == "Dual":
        nl = (
            f"{q.natural_language.rstrip('?')}? "
            f"And how confident are you in this answer?"
        )
        prompt = (
            f"{q.llm_prompt}\n\n"
            f"Also state your confidence level (high/medium/low)."
        )
        return _with_prompt(q, prompt, nl)
    elif level == "Conflicting":
        nl = (
            f"{q.natural_language.rstrip('?')}? "
            f"Or should I be looking at a different metric entirely?"
        )
        prompt = (
            f"{q.llm_prompt}\n\n"
            f"However, if you think a different metric is more appropriate, "
            f"report that instead."
        )
        return _with_prompt(q, prompt, nl)
    return q  # Single = no change


# Factors that are informational only (affect evaluation setup, not question text)
_SETUP_FACTORS = {
    "query_type", "temporal", "hop_count", "reasoning_type",
    "context_scope", "context_order", "context_format",
    "numeric_format", "unit_variation", "precision_hint",
    "memory", "history", "turn_position",
    "tool_availability", "time_pressure", "iteration_limit",
    "context_relevance",
}

# Factor name -> rephrasing function
REPHRASE_FNS: dict[str, Callable] = {
    # Original factors
    "clarity": _rephrase_clarity,
    "length": _rephrase_length,
    "specificity": _rephrase_specificity,
    "style": _rephrase_style,
    "answer_format": _rephrase_answer_format,
    "role": _rephrase_role,
    "expertise": _rephrase_expertise,
    "reasoning_cue": _rephrase_reasoning_cue,
    "entity_aliasing": _rephrase_entity_aliasing,
    "section_hint": _rephrase_section_hint,
    "distractor_content": _rephrase_distractor_content,
    "instruction_conflict": _rephrase_instruction_conflict,
    "answer_anchoring": _rephrase_answer_anchoring,
    "confidence_pressure": _rephrase_confidence_pressure,
    "context_format": _rephrase_context_format,
    "context_scope": _rephrase_context_scope,
    "context_order": _rephrase_context_order,
    # Variability (V) factors
    "paraphrase_depth": _rephrase_paraphrase_depth,
    "granularity": _rephrase_granularity,
    "question_format": _rephrase_question_format,
    # Robustness (R) factors
    "noise": _rephrase_noise,
    "context_relevance": _rephrase_context_relevance,
    "complexity": _rephrase_complexity,
    "edge_case": _rephrase_edge_case,
    "emotional_frame": _rephrase_emotional_frame,
    "format_noise": _rephrase_format_noise,
    "intent_count": _rephrase_intent_count,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _with_prompt(q: GeneratedQuestion, new_prompt: str,
                 new_nl: str | None = None) -> GeneratedQuestion:
    """Return a copy of q with modified llm_prompt (and optionally natural_language)."""
    return GeneratedQuestion(
        qid=q.qid,
        category=q.category,
        natural_language=new_nl or q.natural_language,
        graph_answer_fn=q.graph_answer_fn,   # INVARIANT
        ground_truth=q.ground_truth,          # INVARIANT
        llm_prompt=new_prompt,
        answer_type=q.answer_type,
        metadata={**q.metadata},
    )


def _with_nl(q: GeneratedQuestion, new_nl: str) -> GeneratedQuestion:
    """Return a copy of q with modified natural_language only."""
    return GeneratedQuestion(
        qid=q.qid,
        category=q.category,
        natural_language=new_nl,
        graph_answer_fn=q.graph_answer_fn,   # INVARIANT
        ground_truth=q.ground_truth,          # INVARIANT
        llm_prompt=(q.llm_prompt.replace(q.natural_language, new_nl, 1)
                    if q.metadata.get("assertion_format") else q.llm_prompt),
        answer_type=q.answer_type,
        metadata={**q.metadata},
    )


# ---------------------------------------------------------------------------
# QuestionRephraser
# ---------------------------------------------------------------------------

class QuestionRephraser:
    """Apply DOE factor assignments to rephrase questions.

    Parameters
    ----------
    method : str
        ``"template"`` for deterministic template-based rephrasing.
        ``"llm"`` for LLM-generated rephrasings that preserve the input
        natural-language structure (useful when the base templates already
        read well).
        ``"natural"`` for LLM-generated rephrasings synthesized FROM SEMANTICS
        (head, r1, r2, mid, tail in ``question.metadata``). Bypasses the
        templated ``natural_language`` entirely and forbids graph-query
        vocabulary (node, edge, traverse, hop, relation, ...). Use this when
        the base templates are mechanistic and you want fluent prose.
    llm : knowlytix.core.llm.LLMClient or None
        Required when method="llm" or "natural". Provider-agnostic LLM client
        constructed via ``knowlytix.core.llm.get_llm()``. The client's
        bound model is driven by ``GMS_LLM_MODEL`` (or per-purpose
        overrides); switch providers by setting e.g.
        ``GMS_LLM_MODEL=openai/gpt-4o`` in your environment.
    """

    def __init__(
        self,
        method: str = "template",
        llm=None,
    ):
        if method not in ("template", "llm", "natural"):
            raise ValueError(
                f"method must be 'template', 'llm', or 'natural', got {method!r}"
            )
        if method in ("llm", "natural") and llm is None:
            raise ValueError(
                f"`llm` is required when method={method!r}. "
                "Construct via `from knowlytix.core.llm import get_llm; "
                "llm = get_llm()`."
            )
        self.method = method
        self.llm = llm

        if method == "llm":
            self._templates = _load_prompt_templates()

    def rephrase(
        self,
        question: GeneratedQuestion,
        factor_assignment: dict[str, str],
    ) -> GeneratedQuestion:
        """Apply factor assignment to a single question.

        The ground truth and graph_answer_fn are NEVER modified.
        """
        if question.metadata.get("assertion_format"):
            from .assertions import validate_enrichment

            validate_enrichment(factor_assignment, method=self.method)
        if self.method == "llm":
            return self._rephrase_llm(question, factor_assignment)
        if self.method == "natural":
            return self._rephrase_natural(question, factor_assignment)

        # Template mode
        q = question
        for factor_name, level in factor_assignment.items():
            fn = REPHRASE_FNS.get(factor_name)
            if fn is not None:
                q = fn(q, level)
            # Setup-only factors are recorded but don't modify question text
        return q

    def rephrase_batch(
        self,
        questions: list[GeneratedQuestion],
        assignments: list[dict[str, str]],
    ) -> list[GeneratedQuestion]:
        """Rephrase many (question, assignment) pairs, batching the model call.

        The batched analogue of :meth:`rephrase`, for callers that already hold
        per-item factor assignments (as opposed to a design matrix, which
        :meth:`expand` crosses). Template mode applies the transforms per item
        with no model. LLM and natural modes build every prompt and submit a
        single :meth:`~knowlytix.core.llm.LLMClient.complete_batch` call
        (sequential ``complete`` fallback when the client lacks it), so a local
        engine rephrases the whole set in one forward pass. Results are returned
        aligned to the input order; ``ground_truth`` and ``graph_answer_fn`` are
        preserved for every item.
        """
        if len(questions) != len(assignments):
            raise ValueError(
                f"questions ({len(questions)}) and assignments "
                f"({len(assignments)}) must be the same length")
        for question, assignment in zip(questions, assignments):
            if question.metadata.get("assertion_format"):
                from .assertions import validate_enrichment

                validate_enrichment(assignment, method=self.method)
        if self.method == "template":
            return [self.rephrase(q, a) for q, a in zip(questions, assignments)]

        plan: list[tuple[Any, dict, GeneratedQuestion]] = [
            (i, a, q) for i, (q, a) in enumerate(zip(questions, assignments))]
        rephrased_by_index: dict[int, GeneratedQuestion] = {}
        if self.method == "llm":
            self._batch_fill(plan, rephrased_by_index,
                             self._build_rephrase_messages, self._finish_rephrase,
                             max_tokens=1024, swallow_errors=False)
        else:  # natural
            self._batch_fill(plan, rephrased_by_index,
                             self._build_natural_messages, self._finish_natural,
                             max_tokens=200, swallow_errors=True)
        return [rephrased_by_index[i] for i in range(len(questions))]

    def _build_rephrase_messages(
        self,
        question: GeneratedQuestion,
        factor_assignment: dict[str, str],
    ) -> list[dict[str, str]] | None:
        """Build the chat messages for an LLM rephrase of one question.

        Returns ``None`` when no text-modifying factor applies — the caller
        keeps the original question unchanged and makes no LLM call. Factored
        out of :meth:`_rephrase_llm` so both the single-question path and the
        batched :meth:`expand` path build prompts identically.
        """
        factor_instructions = []
        for factor_name, level in factor_assignment.items():
            instruction = self._templates.get(factor_name, {}).get(level)
            if instruction:
                factor_instructions.append(f"- **{factor_name}={level}**: {instruction}")
            elif factor_name not in _SETUP_FACTORS:
                factor_instructions.append(
                    f"- **{factor_name}={level}**: Apply the '{level}' level for '{factor_name}'."
                )

        if not factor_instructions:
            return None  # No text-modifying factors

        factor_block = "\n".join(factor_instructions)

        prompt = _LLM_REPHRASE_PROMPT.format(
            natural_language=question.natural_language,
            llm_prompt=question.llm_prompt,
            answer_type=question.answer_type,
            ground_truth=question.ground_truth,
            factor_block=factor_block,
        )

        return [
            {"role": "system", "content": _LLM_REPHRASE_SYSTEM},
            {"role": "user", "content": prompt},
        ]

    def _finish_rephrase(
        self,
        question: GeneratedQuestion,
        raw: str,
    ) -> GeneratedQuestion:
        """Parse a raw LLM rephrase response into a ``GeneratedQuestion``.

        Ground truth and ``graph_answer_fn`` are copied unchanged (INVARIANT).
        Shared by the single-question and batched expansion paths.
        """
        rephrased_nl, rephrased_prompt = _parse_llm_rephrase(raw, question)

        return GeneratedQuestion(
            qid=question.qid,
            category=question.category,
            natural_language=rephrased_nl,
            graph_answer_fn=question.graph_answer_fn,   # INVARIANT
            ground_truth=question.ground_truth,          # INVARIANT
            llm_prompt=rephrased_prompt,
            answer_type=question.answer_type,
            metadata={**question.metadata},
        )

    def _rephrase_llm(
        self,
        question: GeneratedQuestion,
        factor_assignment: dict[str, str],
    ) -> GeneratedQuestion:
        """Use an LLM to rephrase the question based on factor assignment.

        The LLM rewrites natural_language and llm_prompt to reflect the
        factor levels naturally. Ground truth and graph_answer_fn are
        copied unchanged. Single-question path; :meth:`expand` batches the
        same prompts across a whole design when the LLM supports it.
        """
        messages = self._build_rephrase_messages(question, factor_assignment)
        if messages is None:
            return question  # No text-modifying factors
        raw = self.llm.complete(messages=messages, max_tokens=1024)
        return self._finish_rephrase(question, raw)

    def _rephrase_natural(
        self,
        question: GeneratedQuestion,
        factor_assignment: dict[str, str],
    ) -> GeneratedQuestion:
        """Synthesize a natural-English question from question.metadata.

        Bypasses the templated ``natural_language`` entirely. Reads the
        semantic structure (source/head, ordered relation chain, expected
        tail) from ``question.metadata`` and writes fluent prose that
        forbids graph-query vocabulary.

        Supported metadata shapes (any one):

        * ``{head, r1, r2, tail, ...}`` — 2-hop transport / multi-hop
        * ``{head, relation, true_tail, ...}`` — 1-hop link-prediction
        * ``{head, relation, ...}`` — generic 1-hop fallback

        Ground truth and graph_answer_fn are copied unchanged. Single-question
        path; :meth:`expand` batches the same prompts across a whole design.
        """
        messages = self._build_natural_messages(question, factor_assignment)
        if messages is None:
            return question  # nothing to synthesize from — original unchanged
        try:
            raw = self.llm.complete(messages=messages, max_tokens=200)
        except Exception:
            raw = ""
        return self._finish_natural(question, raw)

    def _build_natural_messages(
        self,
        question: GeneratedQuestion,
        factor_assignment: dict[str, str],
    ) -> list[dict[str, str]] | None:
        """Build the chat messages for a natural-prose synthesis of one
        question, or ``None`` when ``question.metadata`` lacks a head/relation
        chain to synthesize from (the caller keeps the original unchanged).
        Shared by the single-question and batched paths.
        """
        md = question.metadata or {}
        head = md.get("head") or md.get("source")
        relations: list[str] = []
        if md.get("r1") and md.get("r2"):
            relations = [md["r1"], md["r2"]]
        elif md.get("relation"):
            relations = [md["relation"]]
        if not head or not relations:
            return None

        factor_lines = "\n".join(
            f"  - {k}: {v}" for k, v in factor_assignment.items()
            if v not in (None, "") and k not in _SETUP_FACTORS
        )
        user = (
            f"Source entity: {head}\n"
            f"Property chain (ordered): {list(relations)}\n"
            f"Expected answer (for context only — do NOT include in the "
            f"question): {question.ground_truth}\n"
            f"Answer type: {question.answer_type}\n"
            f"\nFactor assignment for this run:\n{factor_lines}\n"
            f"\nWrite the question now."
        )
        return [
            {"role": "system", "content": _NATURAL_REPHRASE_SYSTEM},
            {"role": "user", "content": user},
        ]

    def _finish_natural(
        self,
        question: GeneratedQuestion,
        raw: str,
    ) -> GeneratedQuestion:
        """Clean a raw natural-prose response into a ``GeneratedQuestion``.

        Out-of-range output (empty, too short, or too long) falls back to the
        templated ``natural_language``. Ground truth and graph_answer_fn are
        copied unchanged. Shared by the single-question and batched paths.
        """
        text = (raw or "").strip().strip('"').strip("'")
        if not (5 < len(text) < 400):
            text = question.natural_language
        return GeneratedQuestion(
            qid=question.qid,
            category=question.category,
            natural_language=text,
            graph_answer_fn=question.graph_answer_fn,   # INVARIANT
            ground_truth=question.ground_truth,          # INVARIANT
            llm_prompt=text,
            answer_type=question.answer_type,
            metadata={**question.metadata},
        )

    def _batch_fill(
        self,
        plan: "list[tuple[Any, dict, GeneratedQuestion]]",
        rephrased_by_index: "dict[int, GeneratedQuestion]",
        build,
        finish,
        max_tokens: int,
        swallow_errors: bool,
    ) -> None:
        """Build a rephrase prompt for every planned (row, question), submit them
        in one ``llm.complete_batch`` call (sequential ``complete`` fallback when
        unavailable), and write each finished question into ``rephrased_by_index``.

        Items whose ``build`` returns ``None`` pass through unchanged.
        ``swallow_errors`` mirrors the natural path's per-item resilience: an LLM
        failure yields an empty response (which ``finish`` maps to a fallback)
        rather than propagating.
        """
        batch_messages: list[list[dict[str, str]]] = []
        batch_targets: list[tuple[int, GeneratedQuestion]] = []
        for i, (_run_idx, assignment, q) in enumerate(plan):
            messages = build(q, assignment)
            if messages is None:
                rephrased_by_index[i] = q
            else:
                batch_messages.append(messages)
                batch_targets.append((i, q))
        if not batch_messages:
            return

        raws = None
        if hasattr(self.llm, "complete_batch"):
            if swallow_errors:
                try:
                    raws = self.llm.complete_batch(batch_messages, max_tokens=max_tokens)
                except Exception:
                    raws = None
            else:
                raws = self.llm.complete_batch(batch_messages, max_tokens=max_tokens)
        if raws is None:  # no batch support, or a swallowed batch failure
            raws = []
            for m in batch_messages:
                if swallow_errors:
                    try:
                        raws.append(self.llm.complete(messages=m, max_tokens=max_tokens))
                    except Exception:
                        raws.append("")
                else:
                    raws.append(self.llm.complete(messages=m, max_tokens=max_tokens))

        for (i, q), raw in zip(batch_targets, raws):
            rephrased_by_index[i] = finish(q, raw)

    def expand(
        self,
        base_questions: list[GeneratedQuestion],
        design_matrix: pd.DataFrame,
    ) -> list[DOEQuestion]:
        """Expand base questions across all design matrix rows.

        Parameters
        ----------
        base_questions : list[GeneratedQuestion]
            Questions from FinStructBench generators.
        design_matrix : pd.DataFrame
            One row per DOE run, columns are factor names.

        Returns
        -------
        list[DOEQuestion]
            len(base_questions) * len(design_matrix) DOE questions,
            each with factor annotations and preserved ground truth.

        Notes
        -----
        In ``method="llm"`` and ``method="natural"`` modes every prompt is built
        up front and submitted in a single
        :meth:`~knowlytix.core.llm.LLMClient.complete_batch` call, so an LLM that
        decodes a batch in one forward pass (a local Hugging Face model, vLLM,
        ...) rephrases the whole design at once instead of one question at a
        time. The call falls back to sequential ``complete`` when the client does
        not implement ``complete_batch``, so any provider works. Output order,
        ``doe_qid`` numbering, and the ground-truth invariants are identical to
        the per-question path.
        """
        # Materialize (run_idx, assignment, question) in final output order.
        plan: list[tuple[Any, dict, GeneratedQuestion]] = []
        for run_idx, row in design_matrix.iterrows():
            assignment = row.to_dict()
            for q in base_questions:
                if q.metadata.get("assertion_format"):
                    from .assertions import validate_enrichment

                    validate_enrichment(assignment, method=self.method)
                plan.append((run_idx, assignment, q))

        rephrased_by_index: dict[int, GeneratedQuestion] = {}
        if self.method == "llm":
            self._batch_fill(plan, rephrased_by_index,
                             self._build_rephrase_messages, self._finish_rephrase,
                             max_tokens=1024, swallow_errors=False)
        elif self.method == "natural":
            self._batch_fill(plan, rephrased_by_index,
                             self._build_natural_messages, self._finish_natural,
                             max_tokens=200, swallow_errors=True)

        doe_questions = []
        for i, (run_idx, assignment, q) in enumerate(plan):
            rephrased = rephrased_by_index.get(i)
            if rephrased is None:
                rephrased = self.rephrase(q, assignment)  # template / natural path
            doe_questions.append(
                DOEQuestion(
                    question=rephrased,
                    factor_assignment=assignment,
                    base_qid=q.qid,
                    doe_qid=f"{q.qid}_run{run_idx:03d}",
                )
            )
        return doe_questions


# ---------------------------------------------------------------------------
# LLM rephrasing prompts and helpers
# ---------------------------------------------------------------------------

_LLM_REPHRASE_SYSTEM = (
    "You are a question rephraser for a DOE (Design of Experiments) testing "
    "framework. Your job is to rephrase questions so they reflect specific "
    "factor levels while preserving the EXACT same answer.\n\n"
    "CRITICAL RULES:\n"
    "1. The correct answer MUST remain exactly the same after rephrasing.\n"
    "2. Do NOT change what information is being asked for.\n"
    "3. Only change HOW the question is phrased, not WHAT it asks.\n"
    "4. The rephrasing must read like natural English a human would actually "
    "ask. NEVER use graph-query vocabulary: node, edge, graph, traverse, "
    "traversal, hop, relation, relationship, entity, link, query, path, "
    "chain, vertex, adjacency. Speak as if the data is just facts about "
    "the world.\n"
    "5. Translate underscored identifiers into natural English. Example: "
    "'data_engineer' -> 'a data engineer'; 'has_risk_tier' -> 'the risk "
    "tier of' / 'what risk tier ... has'. Do not repeat the literal "
    "underscored tokens or quoted relation names from the original.\n"
    "6. Return valid JSON only — no markdown fences, no commentary."
)

_NATURAL_REPHRASE_SYSTEM = (
    "You write ONE natural English question that a real human in the given "
    "role would actually ask. The question must be answerable by following a "
    "chain of properties starting from a source entity.\n"
    "\n"
    "HARD RULES — non-negotiable:\n"
    "1. NEVER use the words: node, edge, graph, traverse, traversal, hop, "
    "   relation, relationship, entity, link, query, path, chain, vertex, "
    "   adjacency. Speak like the data is just facts about the world.\n"
    "2. NEVER repeat identifiers verbatim that contain underscores or look "
    "   like variable names. Translate them to natural English:\n"
    "      data_engineer            -> 'a data engineer' / 'the data engineer'\n"
    "      regulated_lending_model  -> 'a regulated lending model'\n"
    "      model_risk_officer       -> 'the model risk officer'\n"
    "      high_risk_classification -> 'high-risk classification'\n"
    "   And translate relation names to natural phrases:\n"
    "      uses_tool                -> 'the tool ... uses' / 'which tool ... uses'\n"
    "      has_risk_tier            -> 'what risk tier ... has' / 'the risk tier of'\n"
    "      escalates_to             -> 'escalate(s) to' / 'who ... reports to'\n"
    "      has_authority_level      -> 'what authority level' / 'authority level of'\n"
    "      requires_check           -> 'which check is required for' / 'what must be checked'\n"
    "      in_section               -> 'in which section' / 'which section ... is found in'\n"
    "      can_approve              -> 'what can ... approve'\n"
    "      restricted_to_tier       -> 'which tier ... is restricted to'\n"
    "      has_example_fields       -> 'what fields are examples of'\n"
    "   For other relations, derive a natural phrase from the words.\n"
    "3. The question must compose the chain into a single fluent sentence. "
    "   For a 2-hop chain (source -> r1 -> r2 -> ?), the question should ask "
    "   about the second property of the result of the first property — "
    "   not enumerate the steps. Example:\n"
    "      source=data_engineer, relations=[uses_tool, has_risk_tier]\n"
    "      GOOD: 'What risk tier applies to the tool a data engineer uses?'\n"
    "      BAD:  'Follow uses_tool then has_risk_tier from data_engineer.'\n"
    "4. Apply the factor assignment naturally:\n"
    "      role=Auditor + style=Formal      -> precise, professional, may use 'per policy'\n"
    "      role=Employee + style=Casual     -> conversational ('Hey, do you know who ... ?')\n"
    "      role=Regulator + style=Technical -> exact, domain vocabulary\n"
    "      expertise=Novice                 -> avoid jargon\n"
    "      expertise=Expert                 -> use domain terms\n"
    "      length=Short                     -> under 10 words\n"
    "      length=Medium                    -> 10-22 words\n"
    "      length=Long                      -> 22-40 words, one sentence still\n"
    "      query_type=Factual_*             -> 'what / which / who'\n"
    "      query_type=Validation_*          -> 'is it true that' / 'can you confirm that'\n"
    "      query_type=Analytical_Why        -> 'why' phrasing (still answerable from facts)\n"
    "      specificity=Specific             -> precise nouns; specificity=General -> looser\n"
    "5. Output ONLY the rewritten question. No quotes, no preamble, no "
    "   explanation, no JSON. One sentence."
)


_LLM_REPHRASE_PROMPT = """\
Rephrase the following question to reflect the DOE factor levels listed below.

## Original Question
**Natural language:** {natural_language}
**LLM prompt:** {llm_prompt}
**Answer type:** {answer_type}
**Ground truth:** {ground_truth}

## Factor Levels to Apply
{factor_block}

## Instructions
Rewrite both the natural_language (human-readable question) and the llm_prompt \
(the actual prompt sent to the LLM being tested). Apply ALL factor levels \
simultaneously into a single coherent rephrasing. The rephrased question must \
still ask for EXACTLY the same information so that the ground truth answer \
({ground_truth}) remains correct.

Return a JSON object:
{{"rephrased_question": "...", "rephrased_prompt": "..."}}
"""


def _parse_llm_rephrase(
    raw: str, fallback: GeneratedQuestion
) -> tuple[str, str]:
    """Parse LLM rephrase response. Falls back to original on failure."""
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        nl = parsed.get("rephrased_question", "").strip()
        prompt = parsed.get("rephrased_prompt", "").strip()
        if nl and prompt:
            return nl, prompt
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: return originals unchanged
    return fallback.natural_language, fallback.llm_prompt


def _load_prompt_templates() -> dict[str, dict[str, str]]:
    """Load factor instruction templates from prompt_templates.json."""
    template_path = Path(__file__).parent / "prompt_templates.json"
    with open(template_path) as f:
        data = json.load(f)
    return data.get("factor_instructions", {})
