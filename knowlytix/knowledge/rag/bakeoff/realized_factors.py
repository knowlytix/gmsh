# SPDX-License-Identifier: Apache-2.0
"""Realized-factor verification (the DOE manipulation check).

A design matrix assigns each cell an *intended* level for every presentation
factor -- clarity ``Ambiguous``, length ``Long``, and so on. The rewriter that
materializes the cell into text does not always land on the requested level: a
cell meant to be oblique can come back as clean, schema-native phrasing. If the
factor-attributed analysis (which factor level drives a binder to fail) is keyed
on the *requested* coordinate, it is fitted on the wrong axis wherever the
rewriter missed.

This module measures the *realized* level of each factor directly from the
produced text, so downstream analysis keys on the coordinate the text actually
occupies. It is a manipulation check, not a gate -- it never rejects a question,
it relabels it. Measurers are cheap by construction: word-count and token-overlap
for the factors that are a deterministic function of the surface, and a
vectorized nearest-exemplar assignment in the store's embedding space for the
semantic ones (all questions encoded in one batch, not one call each).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

# A per-item measurer maps (question_text, base_text_or_None) -> a level label.
Measurer = Callable[[str, "str | None"], str]

_WORD = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def length_measurer(*, short_max: int = 7, long_min: int = 17,
                    levels: tuple[str, str, str] = ("Short", "Medium", "Long")) -> Measurer:
    """Realized ``length`` by word count: ``<= short_max`` Short, ``>= long_min``
    Long, else Medium."""
    lo, mid, hi = levels

    def measure(question: str, base: str | None = None) -> str:
        n = len(_tokens(question))
        return lo if n <= short_max else (hi if n >= long_min else mid)

    return measure


def length_terciles(questions: list[str],
                    levels: tuple[str, str, str] = ("Short", "Medium", "Long")) -> Measurer:
    """A ``length`` measurer whose Short/Medium/Long bands are the lower and upper
    terciles of ``questions``' own word counts. Length is a relative presentation
    factor, so realizing it is judged against this cohort's own distribution rather
    than absolute word counts that would embed an arbitrary scale."""
    counts = sorted(len(_tokens(q)) for q in questions) or [0]
    n = len(counts)
    lo_cut = counts[max(0, n // 3 - 1)]
    hi_cut = counts[min(n - 1, (2 * n) // 3)]
    return length_measurer(short_max=lo_cut, long_min=max(lo_cut + 1, hi_cut), levels=levels)


def paraphrase_depth_measurer(*, none_min: float = 0.80, light_min: float = 0.45,
                              levels: tuple[str, str, str] = ("None", "Light", "Heavy")
                              ) -> Measurer:
    """Realized ``paraphrase_depth`` by token-set Jaccard against the base
    question: similarity ``>= none_min`` is None (essentially unparaphrased),
    ``>= light_min`` is Light, below is Heavy. Returns Light when no base is
    available rather than guessing an extreme."""
    none_lv, light_lv, heavy_lv = levels

    def measure(question: str, base: str | None = None) -> str:
        if not base:
            return light_lv
        a, b = set(_tokens(question)), set(_tokens(base))
        if not a and not b:
            return none_lv
        jac = len(a & b) / len(a | b) if (a | b) else 1.0
        return none_lv if jac >= none_min else (light_lv if jac >= light_min else heavy_lv)

    return measure


def geometric_level_measurer(levels: dict[str, list[str]], *, encoder=None) -> Measurer:
    """Single-item nearest-exemplar level assignment for a semantic factor.

    Convenience for scoring one question; for a whole cohort prefer
    :meth:`RealizedFactorScorer.annotate`, which encodes all questions in one
    batch. ``levels`` maps each level label to a few exemplar phrases; the text
    is assigned the level whose exemplars it is closest to (cosine in the store's
    embedding space)."""
    assign = _GeometricLevel(levels, encoder=encoder)

    def measure(question: str, base: str | None = None) -> str:
        return assign([question])[0]

    return measure


class _GeometricLevel:
    """Vectorized nearest-exemplar level assigner. Encodes level exemplars once;
    scores a batch of questions against them in a single encode call."""

    def __init__(self, levels: dict[str, list[str]], *, encoder=None):
        if not levels:
            raise ValueError("geometric level assigner needs at least one level")
        self.levels = list(levels)
        if encoder is None:
            from knowlytix.core.graph.encoders import encode_texts
            encoder = encode_texts
        self.encoder = encoder
        owners, phrases = [], []
        for lab, exs in levels.items():
            for ex in exs:
                owners.append(lab)
                phrases.append(ex)
        self._owners = owners
        self._ex = self._encode(phrases)  # (n_ex, dim), normalized

    def _encode(self, texts: list[str]):
        import torch
        emb = torch.as_tensor(self.encoder(list(texts)), dtype=torch.float32)
        return torch.nn.functional.normalize(emb, p=2, dim=-1)

    def __call__(self, questions: list[str]) -> list[str]:
        if not questions:
            return []
        q = self._encode(questions)                 # (n_q, dim)
        sim = q @ self._ex.T                         # (n_q, n_ex) cosine
        out = []
        # per-level max over its exemplars, then argmax level
        for i in range(sim.shape[0]):
            best_lab, best = self.levels[0], float("-inf")
            per: dict[str, float] = {}
            row = sim[i]
            for j, lab in enumerate(self._owners):
                v = float(row[j])
                if v > per.get(lab, float("-inf")):
                    per[lab] = v
            for lab in self.levels:
                if per.get(lab, float("-inf")) > best:
                    best, best_lab = per[lab], lab
            out.append(best_lab)
        return out


# Default exemplar banks for the standard presentation factors. Deliberately
# short and generic -- this is a cheap relabeling signal, not a trained model.
_DEFAULT_SEMANTIC_LEVELS: dict[str, dict[str, list[str]]] = {
    "clarity": {
        "Clear": ["What was the total revenue for the company in 2024?",
                  "State the reported net income.",
                  "Which auditor signed the annual report?"],
        "Ambiguous": ["What about the numbers for that period?",
                      "How did things look financially?",
                      "Can you say something regarding the figures?"],
        "Misleading": ["Given the loss last year, what was the 2024 revenue?",
                       "Since the auditor resigned, who signed off?",
                       "After the restatement, what was net income?"],
    },
    "style": {
        "Formal": ["Kindly provide the reported total revenue for fiscal year 2024.",
                   "Please state the net income as disclosed in the annual report."],
        "Casual": ["Hey, what'd they make in revenue in 2024?",
                   "So how much profit did they pull in?"],
        "Technical": ["Report FY2024 consolidated total revenue (USD, as-reported).",
                      "Return net income attributable to shareholders, FY2024."],
    },
    "expertise": {
        "Novice": ["What is revenue and how much did the company make?",
                   "I'm new to this -- what were the earnings?"],
        "Intermediate": ["What was total revenue and net income for FY2024?",
                         "How did operating margin move year over year?"],
        "Expert": ["Decompose FY2024 revenue recognition and its effect on operating margin.",
                   "Reconcile reported net income to adjusted EBITDA for FY2024."],
    },
}


@dataclass
class RealizedFactorScorer:
    """Measure realized factor levels for the questions in a DOE cohort.

    Parameters
    ----------
    item_measurers:
        ``{factor: measurer}`` for factors that are a deterministic per-item
        function of the surface (length, paraphrase depth).
    semantic_levels:
        ``{factor: {level: [exemplar, ...]}}`` for factors assigned by
        nearest exemplar in the embedding space, encoded in one batch.
    base_lookup:
        ``{seed: base_question_text}`` so surface-comparison measurers reach the
        unparaphrased base. Optional.
    encoder:
        ``list[str] -> Tensor`` of row embeddings; defaults to the graph's
        ``encode_texts``. Pass the store's document-tuned encoder to score in the
        same space retrieval uses.
    """

    item_measurers: dict[str, Measurer] = field(default_factory=dict)
    semantic_levels: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    base_lookup: dict[str, str] = field(default_factory=dict)
    encoder: object = None
    _assigners: dict[str, _GeometricLevel] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._assigners = {f: _GeometricLevel(lv, encoder=self.encoder)
                           for f, lv in self.semantic_levels.items()}

    @classmethod
    def default(cls, *, base_lookup: dict[str, str] | None = None, encoder=None,
                semantic_levels: dict[str, dict[str, list[str]]] | None = None
                ) -> "RealizedFactorScorer":
        """Standard scorer for the clarity/style/length/expertise/paraphrase_depth
        catalog: deterministic length + paraphrase depth, geometric assignment for
        the semantic factors."""
        return cls(
            item_measurers={"length": length_measurer(),
                            "paraphrase_depth": paraphrase_depth_measurer()},
            semantic_levels=semantic_levels or _DEFAULT_SEMANTIC_LEVELS,
            base_lookup=dict(base_lookup or {}),
            encoder=encoder,
        )

    @property
    def factors(self) -> list[str]:
        return list(self.item_measurers) + list(self.semantic_levels)

    def calibrate_length(self, cohort: list[dict], *, question_key: str = "question",
                         factor: str = "length") -> "RealizedFactorScorer":
        """Rebind the ``length`` measurer to this cohort's word-count terciles, so
        realized length is judged on the cohort's own scale. No-op if ``length`` is
        not a measured factor. Returns self for chaining."""
        if factor in self.item_measurers:
            qs = [r.get(question_key, "") for r in cohort]
            self.item_measurers[factor] = length_terciles(qs)
        return self

    def measure(self, question: str, *, base: str | None = None,
                seed: str | None = None) -> dict[str, str]:
        """Realized level per measured factor for one question."""
        if base is None and seed is not None:
            base = self.base_lookup.get(seed)
        out = {f: m(question, base) for f, m in self.item_measurers.items()}
        for f, assign in self._assigners.items():
            out[f] = assign([question])[0]
        return out

    def annotate(self, cohort: list[dict], *, question_key: str = "question",
                 seed_key: str = "_seed", out_key: str = "_realized") -> list[dict]:
        """Return a copy of ``cohort`` with each row carrying ``out_key`` = the
        realized levels. All questions are batch-encoded once per semantic
        factor."""
        rows = [dict(r) for r in cohort]
        questions = [r.get(question_key, "") for r in rows]
        realized: list[dict[str, str]] = [{} for _ in rows]
        # deterministic per-item factors
        for f, m in self.item_measurers.items():
            for i, r in enumerate(rows):
                base = self.base_lookup.get(r.get(seed_key))
                realized[i][f] = m(questions[i], base)
        # semantic factors, batched
        for f, assign in self._assigners.items():
            labels = assign(questions)
            for i, lab in enumerate(labels):
                realized[i][f] = lab
        for i, r in enumerate(rows):
            r[out_key] = realized[i]
        return rows

    def manipulation_report(self, cohort: list[dict], *, intended_key: str = "_factors",
                            realized_key: str = "_realized",
                            question_key: str = "question", seed_key: str = "_seed"
                            ) -> dict:
        """Per-factor manipulation check: how often the realized level matched the
        intended one, and the intended->realized confusion. A low ``match_rate``
        means the rewriter frequently missed that factor's requested level -- the
        signal that keying analysis on the intended coordinate would be wrong."""
        rows = cohort
        if not rows or realized_key not in rows[0]:
            rows = self.annotate(cohort, question_key=question_key, seed_key=seed_key,
                                  out_key=realized_key)
        report: dict[str, dict] = {}
        for factor in self.factors:
            confusion: dict[str, dict[str, int]] = {}
            match = n = 0
            for row in rows:
                intended = (row.get(intended_key) or {}).get(factor)
                realized = (row.get(realized_key) or {}).get(factor)
                if intended is None or realized is None:
                    continue
                n += 1
                match += int(intended == realized)
                confusion.setdefault(intended, {}).setdefault(realized, 0)
                confusion[intended][realized] += 1
            report[factor] = {
                "n": n,
                "match_rate": round(match / n, 4) if n else 0.0,
                "confusion": confusion,
                # "surface" measurers are deterministic functions of the text;
                # "semantic" ones are a cheap embedding classifier whose match_rate
                # reflects classifier fidelity as well as rewriter drift.
                "kind": "surface" if factor in self.item_measurers else "semantic",
            }
        return report
