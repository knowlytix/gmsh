# SPDX-License-Identifier: Apache-2.0
"""
DOE Benchmark Orchestrator.

Composes FinStructBench's Benchmark (graph ingestion + question generation)
with DOE design matrix expansion and any evaluator (LLM or custom agent).
"""

import re
import time
from typing import Callable

import numpy as np
import yaml

from knowlytix.bench.benchmark import Benchmark
from knowlytix.bench.ingest import parse_sections
from knowlytix.bench.scorers import score_answer, PARSERS
from knowlytix.bench.llm_caller import call_llm

from .design_matrix import DesignMatrix
from .factor_loader import FactorLoader
from .question_rephraser import QuestionRephraser, DOEQuestion
from .results import DOEBenchmarkResult, DOEQuestionResult
from .analysis import DOEAnalyzer


# ── Context extraction helpers ───────────────────────────────────────


def _normalize_tag(section_name: str) -> str:
    """Convert section heading to the tag used by ingest.py."""
    tag = re.sub(r"^\d+\.\s*", "", section_name).strip()
    tag = re.sub(r"[^a-zA-Z0-9_\s]", "", tag).strip()
    return tag.replace(" ", "_").lower()


def _build_section_map(markdown: str) -> dict[str, str]:
    """Build tag -> section_text mapping.

    Merges subsections into their parent so that a tag like
    'capital_adequacy_ratios' returns all subsection tables too.
    """
    raw_sections = parse_sections(markdown)
    tag_map: dict[str, list[str]] = {}

    for key, content in raw_sections.items():
        # "2. Capital Composition/Table 2.1 — CET1" → parent = "2. Capital Composition"
        parent = key.split("/")[0]
        tag = _normalize_tag(parent)
        if tag not in tag_map:
            tag_map[tag] = []
        if content:
            heading = key.split("/")[-1] if "/" in key else key
            tag_map[tag].append(f"### {heading}\n{content}")

    return {tag: "\n\n".join(parts) for tag, parts in tag_map.items()}


def _section_for_question(dq: DOEQuestion, graph) -> str | None:
    """Infer the section tag for a DOE question from its metadata."""
    meta = dq.question.metadata

    # exact_recall, multi_hop: have enm_type = section tag
    if "enm_type" in meta:
        return meta["enm_type"]

    # multi_hop cross-type: has type1, type2
    if "type1" in meta:
        return meta["type1"]

    # threshold: extract section from the llm_prompt which says
    # "in the '<cat>' section"
    if "encoder" in meta:
        m = re.search(r"in the '([^']+)' section", dq.question.llm_prompt)
        if m:
            return m.group(1)

    # cross_reference: find sections via relation triples
    for rel_key in ("relation_1", "relation_2"):
        if rel_key in meta:
            triples = graph.query_triples(relation=meta[rel_key])
            for h, r, t in triples:
                sec_triples = graph.query_triples(head=h, relation="in_section")
                if sec_triples:
                    return sec_triples[0][2]

    # counting: find via relation + optional tail
    if "relation" in meta:
        kwargs = {"relation": meta["relation"]}
        if "tail" in meta:
            kwargs["tail"] = meta["tail"]
        triples = graph.query_triples(**kwargs)
        for h, r, t in triples:
            sec_triples = graph.query_triples(head=h, relation="in_section")
            if sec_triples:
                return sec_triples[0][2]

    return None


def _get_context(
    dq: DOEQuestion,
    full_markdown: str,
    section_map: dict[str, str],
    graph,
    context_mode: str,
    rng: np.random.Generator,
) -> str:
    """Return the context string based on context_mode."""
    if context_mode == "full":
        return full_markdown

    tag = _section_for_question(dq, graph)

    if tag is None or tag not in section_map:
        # Fallback to full document if section can't be determined
        return full_markdown

    relevant = section_map[tag]

    if context_mode == "oracle_section":
        return relevant

    if context_mode == "noisy_section":
        # Add 1-2 random irrelevant sections as distractors
        other_tags = [t for t in section_map if t != tag]
        n_distractors = min(rng.integers(1, 3), len(other_tags))
        distractor_tags = rng.choice(other_tags, size=n_distractors, replace=False)
        parts = [relevant] + [section_map[t] for t in distractor_tags]
        rng.shuffle(parts)
        return "\n\n---\n\n".join(parts)

    return full_markdown


class DOEBenchmark:
    """Graph-verified DOE testing for LLMs and agentic systems.

    Parameters
    ----------
    markdown_path : str
        Path to the document (markdown).
    factor_group : str or None
        Predefined factor group to use.
    factor_names : list[str] or None
        Explicit factor names (combined with factor_group).
    n_runs : int
        Number of DOE design points.
    max_per_category : int
        Max base questions per category from FinStructBench.
    method : str
        DOE sampling method: "sobol", "lhs", "grid".
    seed : int
        Random seed.
    catalog_path : str or None
        Path to factor catalog. None uses bundled catalog.
    rephrase_method : str
        "template" for deterministic rephrasing,
        "llm" for LLM-generated natural rephrasings.
    rephrase_llm : knowlytix.core.llm.LLMClient or None
        Required when rephrase_method="llm". Provider-agnostic client
        constructed via ``knowlytix.core.llm.get_llm()``. The client's
        bound model is driven by ``GMS_LLM_MODEL`` (default resolves
        to ``anthropic/claude-sonnet-4-20250514``); switch providers
        by setting e.g. ``GMS_LLM_MODEL=openai/gpt-4o``.
    """

    def __init__(
        self,
        markdown_path: str,
        factor_group: str | None = None,
        factor_names: list[str] | None = None,
        n_runs: int = 32,
        max_per_category: int = 5,
        method: str = "sobol",
        seed: int = 42,
        catalog_path: str | None = None,
        rephrase_method: str = "template",
        rephrase_llm=None,
    ):
        self.markdown_path = markdown_path
        self.max_per_category = max_per_category
        self.seed = seed

        # Build the FinStructBench base benchmark
        self._base = Benchmark(
            markdown_path,
            max_per_category=max_per_category,
            seed=seed,
        )

        # Load factors and build design matrix
        loader = FactorLoader(catalog_path)
        config = {}
        if factor_group:
            config["use_group"] = factor_group
        if factor_names:
            config["use_factors"] = factor_names
        if not config:
            config["use_group"] = "quick_screen"

        self.factors = loader.load_factors(config)
        loader.validate_factors(self.factors)

        self.design_matrix = DesignMatrix(
            self.factors, method=method, n_runs=n_runs, seed=seed
        )
        self.rephraser = QuestionRephraser(
            method=rephrase_method,
            llm=rephrase_llm,
        )

    @classmethod
    def from_yaml(cls, config_path: str, rephrase_llm=None) -> "DOEBenchmark":
        """Create DOEBenchmark from YAML configuration."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        exp = cfg.get("experiment", {})
        doc = cfg.get("document", {})
        factors_cfg = cfg.get("factors", {})
        dm_cfg = cfg.get("design_matrix", {})
        rephrase_cfg = cfg.get("rephrasing", {})

        return cls(
            markdown_path=doc["source"],
            factor_group=factors_cfg.get("use_group"),
            factor_names=factors_cfg.get("use_factors"),
            n_runs=dm_cfg.get("n_runs", exp.get("n_design_runs", 32)),
            max_per_category=exp.get("max_per_category", 5),
            method=dm_cfg.get("method", "sobol"),
            seed=dm_cfg.get("seed", exp.get("seed", 42)),
            catalog_path=factors_cfg.get("catalog_path"),
            rephrase_method=rephrase_cfg.get("method", "template"),
            rephrase_llm=rephrase_llm,
        )

    def generate_questions(self) -> list[DOEQuestion]:
        """Generate DOE-expanded questions.

        1. Generate base questions from FinStructBench generators
        2. Generate design matrix
        3. Expand: base_questions x design_matrix -> DOE questions
        """
        base_qs = self._base.generate_questions()
        dm = self.design_matrix.generate()

        print(f"\nDOE expansion: {len(base_qs)} base x {len(dm)} runs "
              f"= {len(base_qs) * len(dm)} questions")
        print(f"  Factors: {[f['name'] for f in self.factors]}")

        return self.rephraser.expand(base_qs, dm)

    def run(
        self,
        llm=None,
        evaluator: Callable[[str, str], str] | None = None,
        context_mode: str = "full",
    ) -> DOEBenchmarkResult:
        """Run the DOE benchmark.

        Provide exactly one of:
        - llm: provider-agnostic ``knowlytix.core.llm.LLMClient``
               (construct via ``knowlytix.core.llm.get_llm()``)
        - evaluator: callable(question, context) -> answer string

        Parameters
        ----------
        llm : knowlytix.core.llm.LLMClient or None
            Provider-agnostic LLM client. Its bound model is driven
            by ``GMS_LLM_MODEL`` (default anthropic/claude-sonnet-4-...);
            switch providers by setting e.g. ``GMS_LLM_MODEL=openai/gpt-4o``.
        evaluator : callable or None
            Custom evaluator: f(question: str, context: str) -> str.
        context_mode : str
            How much document context to provide:
            - "full": entire document (default, current behavior)
            - "oracle_section": only the section relevant to the question
            - "noisy_section": relevant section + 1-2 random distractors
        """
        if llm is None and evaluator is None:
            raise ValueError("Provide llm or evaluator")
        if context_mode not in ("full", "oracle_section", "noisy_section"):
            raise ValueError(
                f"context_mode must be 'full', 'oracle_section', or "
                f"'noisy_section', got '{context_mode}'"
            )

        doe_questions = self.generate_questions()
        section_map = _build_section_map(self._base.markdown)
        ctx_rng = np.random.default_rng(self.seed)

        # Model label: from the LLMClient's bound model, or a sentinel
        # for custom evaluators.
        model_label = getattr(llm, "model", "llm") if llm else "custom_evaluator"

        result = DOEBenchmarkResult(
            instance_name=self._base.instance_name,
            total_questions=len(doe_questions),
            n_base_questions=len(set(dq.base_qid for dq in doe_questions)),
            n_design_runs=self.design_matrix.n_runs,
            factors=[f["name"] for f in self.factors],
            model=model_label,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            graph_stats=self._base.graph.stats(),
        )
        result.context_mode = context_mode

        print(f"\nEvaluating {len(doe_questions)} DOE questions "
              f"(context_mode={context_mode})...")
        for i, dq in enumerate(doe_questions):
            q = dq.question
            tag = f"[{i + 1}/{len(doe_questions)}] {dq.doe_qid}"

            try:
                print(f"  {tag}...", end=" ", flush=True)

                ctx = _get_context(
                    dq, self._base.markdown, section_map,
                    self._base.graph, context_mode, ctx_rng,
                )

                if evaluator is not None:
                    raw = evaluator(q.llm_prompt, ctx)
                else:
                    # call_llm duck-types: any object with a `.complete()`
                    # method (our LLMClient) routes through the provider-
                    # agnostic path; the `model=None` kwarg lets the
                    # client's bound model drive the request.
                    raw = call_llm(llm, ctx, q.llm_prompt, None)

                parser = PARSERS.get(q.answer_type, PARSERS["str"])
                answer = parser(raw)
                score = score_answer(answer, q.ground_truth)

                if score.correct:
                    result.total_correct += 1

                qr = DOEQuestionResult(
                    doe_qid=dq.doe_qid,
                    base_qid=dq.base_qid,
                    category=q.category,
                    question=q.natural_language,
                    ground_truth=str(q.ground_truth),
                    answer=str(answer),
                    correct=score.correct,
                    detail=score.detail,
                    factor_assignment=dq.factor_assignment,
                    raw_response=raw[:500] if raw else "",
                )
                status = "PASS" if score.correct else "FAIL"
                print(status)

            except Exception as e:
                qr = DOEQuestionResult(
                    doe_qid=dq.doe_qid,
                    base_qid=dq.base_qid,
                    category=q.category,
                    question=q.natural_language,
                    ground_truth=str(q.ground_truth),
                    answer=f"ERROR: {e}",
                    correct=False,
                    detail=f"error: {e}",
                    factor_assignment=dq.factor_assignment,
                )
                print(f"ERROR: {e}")

            result.questions.append(qr)

        return result

    def run_graph_only(self) -> DOEBenchmarkResult:
        """Run graph baseline only (no LLM). Validates ground truth invariance."""
        doe_questions = self.generate_questions()

        result = DOEBenchmarkResult(
            instance_name=self._base.instance_name,
            total_questions=len(doe_questions),
            n_base_questions=len(set(dq.base_qid for dq in doe_questions)),
            n_design_runs=self.design_matrix.n_runs,
            factors=[f["name"] for f in self.factors],
            model="graph_baseline",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            graph_stats=self._base.graph.stats(),
        )

        print(f"\nGraph baseline: {len(doe_questions)} DOE questions...")
        for dq in doe_questions:
            q = dq.question
            graph_answer = q.graph_answer_fn(self._base.graph)
            score = score_answer(graph_answer, q.ground_truth)

            if score.correct:
                result.total_correct += 1

            result.questions.append(
                DOEQuestionResult(
                    doe_qid=dq.doe_qid,
                    base_qid=dq.base_qid,
                    category=q.category,
                    question=q.natural_language,
                    ground_truth=str(q.ground_truth),
                    answer=str(graph_answer),
                    correct=score.correct,
                    detail=score.detail,
                    factor_assignment=dq.factor_assignment,
                )
            )

        print(f"  Graph baseline: {result.total_correct}/{result.total_questions} "
              f"({result.accuracy:.0%})")
        if result.total_correct != result.total_questions:
            print("  WARNING: Graph baseline is not 100% — ground truth issue!")

        return result

    def analyze(self, result: DOEBenchmarkResult) -> DOEAnalyzer:
        """Create analyzer from results."""
        return DOEAnalyzer(result)

    @staticmethod
    def print_results(result: DOEBenchmarkResult):
        """Print summary of DOE benchmark results."""
        n = result.total_questions
        print(f"\n{'=' * 70}")
        print(f"DOE BENCHMARK — {result.instance_name}")
        print(f"{'=' * 70}")
        print(f"  Model:          {result.model}")
        print(f"  Base questions:  {result.n_base_questions}")
        print(f"  Design runs:    {result.n_design_runs}")
        print(f"  Total questions: {n}")
        print(f"  Factors:         {result.factors}")
        print(f"  Accuracy:        {result.total_correct}/{n} ({result.accuracy:.1%})")

        # Per-category
        df = result.to_dataframe()
        if not df.empty:
            print(f"\n  {'Category':<20} {'Correct':>8} {'Total':>8} {'Rate':>8}")
            print(f"  {'─' * 48}")
            for cat in sorted(df["category"].unique()):
                sub = df[df["category"] == cat]
                correct = sub["correct"].sum()
                total = len(sub)
                print(f"  {cat:<20} {correct:>8} {total:>8} {correct/total:>7.0%}")
