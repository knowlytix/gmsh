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
from typing import Callable

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
        # Add verbose context
        nl = (
            f"Given the information presented in the document, and considering "
            f"all relevant data points, {q.natural_language.lower()} "
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


def _rephrase_style(q: GeneratedQuestion, level: str) -> GeneratedQuestion:
    """Modify communication style."""
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
        llm_prompt=q.llm_prompt,
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
        "template" for deterministic template-based rephrasing.
        "llm" for LLM-generated natural rephrasings.
    llm : knowlytix.core.llm.LLMClient or None
        Required when method="llm". Provider-agnostic LLM client
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
        if method not in ("template", "llm"):
            raise ValueError(f"method must be 'template' or 'llm', got {method!r}")
        if method == "llm" and llm is None:
            raise ValueError(
                "`llm` is required when method='llm'. "
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
        if self.method == "llm":
            return self._rephrase_llm(question, factor_assignment)

        # Template mode
        q = question
        for factor_name, level in factor_assignment.items():
            fn = REPHRASE_FNS.get(factor_name)
            if fn is not None:
                q = fn(q, level)
            # Setup-only factors are recorded but don't modify question text
        return q

    def _rephrase_llm(
        self,
        question: GeneratedQuestion,
        factor_assignment: dict[str, str],
    ) -> GeneratedQuestion:
        """Use an LLM to rephrase the question based on factor assignment.

        The LLM rewrites natural_language and llm_prompt to reflect the
        factor levels naturally. Ground truth and graph_answer_fn are
        copied unchanged.
        """
        # Build factor instruction block
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
            return question  # No text-modifying factors

        factor_block = "\n".join(factor_instructions)

        prompt = _LLM_REPHRASE_PROMPT.format(
            natural_language=question.natural_language,
            llm_prompt=question.llm_prompt,
            answer_type=question.answer_type,
            ground_truth=question.ground_truth,
            factor_block=factor_block,
        )

        raw = self.llm.complete(
            messages=[
                {"role": "system", "content": _LLM_REPHRASE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
        )

        # Parse JSON response
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
        """
        doe_questions = []
        for run_idx, row in design_matrix.iterrows():
            assignment = row.to_dict()
            for q in base_questions:
                rephrased = self.rephrase(q, assignment)
                doe_q = DOEQuestion(
                    question=rephrased,
                    factor_assignment=assignment,
                    base_qid=q.qid,
                    doe_qid=f"{q.qid}_run{run_idx:03d}",
                )
                doe_questions.append(doe_q)
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
    "4. Return valid JSON only — no markdown fences, no commentary."
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
