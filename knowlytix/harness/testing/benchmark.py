# SPDX-License-Identifier: Apache-2.0
"""DOE-GMS Benchmark Orchestrator.

End-to-end pipeline:
1. Ingest markdown → GMS (ground truth)
2. Generate base questions from GMS via eval generators
3. Expand via DOE design matrix (graphDOE)
4. Run against target (LLM or agent)
5. Score using GMS ground truth + eval scorers
6. Trace every event to DOETracingGMS
7. Analyze with DOEAnalyzer (logistic regression)
"""

from __future__ import annotations

import os
import re
import time
import uuid
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .verdict import TypedVerdict

import numpy as np
import torch

from knowlytix.core.llm import LLMSettings, ModelPurpose
from knowlytix.knowledge.config import DocGMSConfig
from knowlytix.knowledge.ingest import ingest_document
from knowlytix.knowledge.llm_backend import LiteLLMBackend
from knowlytix.knowledge.store import GMSExpertStore

from knowlytix.benchmark.eval.scorers import score_answer, PARSERS

# graphdoe lives under knowlytix.harness.graphdoe as of the 2026-04-20
# upstream sync + Phase-4.8 namespace alignment. The try/except stub
# previously used here (to survive a missing external graphdoe dep) is
# retired: graphdoe now ships with the harness package, so these imports
# always succeed wherever `knowlytix-harness` is installed.
from knowlytix.harness.graphdoe.design_matrix import DesignMatrix
from knowlytix.harness.graphdoe.factor_loader import FactorLoader
from knowlytix.harness.graphdoe.question_rephraser import (
    QuestionRephraser,
    DOEQuestion,
)
from knowlytix.harness.graphdoe.results import (
    DOEBenchmarkResult,
    DOEQuestionResult,
)
from knowlytix.harness.graphdoe.analysis import DOEAnalyzer

from .bridge import generate_base_questions
from .config import DOEHarnessConfig
from .judge import GMSJudge, GeometricVerdict
from .tracing import DOETracingGMS


# ── Context extraction helpers (adapted from graphDOE) ─────────────────


def _normalize_tag(section_name: str) -> str:
    tag = re.sub(r"^\d+\.\s*", "", section_name).strip()
    tag = re.sub(r"[^a-zA-Z0-9_\s]", "", tag).strip()
    return tag.replace(" ", "_").lower()


def _build_section_map(markdown: str) -> dict[str, str]:
    """Build tag -> section_text mapping from markdown headings."""
    sections: dict[str, list[str]] = {}
    current_tag = "_preamble"
    current_lines: list[str] = []

    for line in markdown.split("\n"):
        heading_match = re.match(r"^(#{1,3})\s+(.+)", line)
        if heading_match:
            # Save previous
            if current_lines:
                sections.setdefault(current_tag, []).append(
                    "\n".join(current_lines))
            current_tag = _normalize_tag(heading_match.group(2))
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.setdefault(current_tag, []).append("\n".join(current_lines))

    return {tag: "\n\n".join(parts) for tag, parts in sections.items()}


def _section_for_question(dq: DOEQuestion) -> str | None:
    """Infer the section tag for a DOE question from its metadata."""
    meta = dq.question.metadata
    if "enm_type" in meta:
        return meta["enm_type"]
    if "type1" in meta:
        return meta["type1"]
    if "encoder" in meta:
        m = re.search(r"in the '([^']+)' section", dq.question.llm_prompt)
        if m:
            return m.group(1)
    return None


def _get_context(
    dq: DOEQuestion,
    full_markdown: str,
    section_map: dict[str, str],
    context_mode: str,
    rng: np.random.Generator,
) -> str:
    if context_mode == "full":
        return full_markdown

    tag = _section_for_question(dq)
    if tag is None or tag not in section_map:
        return full_markdown

    relevant = section_map[tag]

    if context_mode == "oracle_section":
        return relevant

    if context_mode == "noisy_section":
        other_tags = [t for t in section_map if t != tag]
        n_distractors = min(rng.integers(1, 3), len(other_tags))
        if n_distractors > 0:
            distractor_tags = rng.choice(
                other_tags, size=n_distractors, replace=False)
            parts = [relevant] + [section_map[t] for t in distractor_tags]
            rng.shuffle(parts)
            return "\n\n---\n\n".join(parts)
        return relevant

    return full_markdown


# ── Main Orchestrator ──────────────────────────────────────────────────


class DOEGMSBenchmark:
    """DOE benchmark that uses GMS as ground truth and judge.

    Usage::

        config = DOEHarnessConfig(markdown_path="report.md")
        bench = DOEGMSBenchmark(config)
        bench.ingest()
        result = bench.run()
        analyzer = bench.analyze(result)
        analyzer.print_report()
    """

    def __init__(self, config: DOEHarnessConfig):
        self.config = config
        self.device = torch.device(
            config.device if config.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.store: GMSExpertStore | None = None
        self.markdown: str = ""
        self.tracing: DOETracingGMS | None = None
        self.judge: GMSJudge | None = None
        self._doe_questions: list[DOEQuestion] | None = None

        # GMSH Phase 1: typed verdicts
        self._typed_verdicts: dict[str, "TypedVerdict"] = {}

        # GMSH Phase 2: provenance and truth
        self.provenance = None
        self.truth_store = None

        # GMSH Phase 4: snapshots
        self.snapshots = None

        # Load DOE factors
        loader = FactorLoader()
        factor_config: dict = {}
        if config.factor_group:
            factor_config["use_group"] = config.factor_group
        if config.factor_names:
            factor_config["use_factors"] = config.factor_names
        if not factor_config:
            factor_config["use_group"] = "quick_screen"

        self.factors = loader.load_factors(factor_config)
        loader.validate_factors(self.factors)

        self.design_matrix = DesignMatrix(
            self.factors,
            method=config.method,
            n_runs=config.n_runs,
            seed=config.seed,
        )
        self.rephraser = QuestionRephraser(method=config.rephrase_method)

        # Initialize tracing
        if config.enable_tracing:
            self.tracing = DOETracingGMS(config.tracing_dir, self.device)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DOEGMSBenchmark":
        """Create benchmark from YAML config file."""
        config = DOEHarnessConfig.from_yaml(yaml_path)
        return cls(config)

    # ----- Step 1: Ingest markdown → GMS ----------------------------------

    def ingest(self, markdown_path: str | None = None) -> GMSExpertStore:
        """Convert markdown to GMS store (ground truth).

        If a store already exists for this document, loads it instead.
        """
        md_path = markdown_path or self.config.markdown_path
        if not md_path:
            raise ValueError("No markdown_path provided")

        store_name = self.config.resolve_store_name()
        store_path = os.path.join(self.config.stores_dir, store_name)

        doc_config = DocGMSConfig(store_path=store_path)
        doc_config.train.epochs = self.config.ingest_epochs
        self.store = GMSExpertStore(doc_config, self.device)

        t0_ingest = time.time()
        loaded_existing = self.store.exists()
        if loaded_existing:
            print(f"  Loading existing GMS store: {store_path}")
            self.store.load()
        else:
            print(f"  Ingesting {md_path} → {store_path}")
            # Provider-agnostic: GMS_LLM_MODEL env var picks the provider
            # (anthropic/openai/google/ollama/…); default resolves to
            # anthropic/claude-sonnet-4-20250514 via LLMSettings.DEFAULT_MODEL
            # for zero-config out-of-box use.
            llm_model = LLMSettings().resolve(ModelPurpose.DEFAULT)
            llm = LiteLLMBackend(llm_model)
            ingest_document(self.store, md_path, llm, doc_config, self.device)
        ingest_elapsed = time.time() - t0_ingest

        # Load markdown for context extraction
        with open(md_path) as f:
            self.markdown = f.read()

        # Create GMS geometric judge
        self.judge = GMSJudge(
            self.store,
            semantic_mode=self.config.semantic_mode,
            semantic_model=(
                self.config.semantic_model or self.config.target_model
            ),
            tracing=self.tracing,
        )

        stats = self.store.stats()
        print(f"  GMS store: {stats['entities']} entities, "
              f"{stats['triples']} triples, {stats['enm_entries']} ENM")
        print("  GMS judge: geometric scoring enabled "
              "(geodesic + tension + holonomy)")

        # Trace ingestion event
        if self.tracing:
            self.tracing.trace_ingest(
                store_name=store_name,
                markdown_path=md_path,
                loaded_existing=loaded_existing,
                elapsed=ingest_elapsed,
                store_stats=stats,
            )

        # GMSH Phase 2: Provenance and truth store
        if self.config.enable_provenance:
            from .provenance import ProvenanceRegistry
            from .truth import TruthStore

            self.provenance = ProvenanceRegistry(self.tracing)
            self.provenance.register_document(md_path, self.markdown)

            self.truth_store = TruthStore(self.store, tracing=self.tracing)
            self.truth_store.build_from_store()

            # Save truth store alongside GMS store
            truth_path = os.path.join(store_path, "truth_store.json")
            self.truth_store.save(truth_path)

            # Wire truth store into claim compiler if typed verdicts enabled
            if (self.config.enable_typed_verdicts
                    and self.judge
                    and hasattr(self.judge, "_compiler")):
                self.judge._compiler.truth_store = self.truth_store

        # GMSH Phase 4: Take snapshot after ingest
        if self.config.snapshot_dir:
            from .snapshots import SnapshotManager

            self.snapshots = SnapshotManager(
                self.config.snapshot_dir, self.tracing)
            self.snapshots.take_snapshot(
                self.store, run_id="ingest", trigger="ingest")

        return self.store

    # ----- Step 2: Generate DOE questions ---------------------------------

    def generate_questions(self) -> list[DOEQuestion]:
        """Generate base questions from GMS, then expand via DOE."""
        if self.store is None:
            raise RuntimeError("Call ingest() first")

        base_qs = generate_base_questions(
            self.store,
            generator_names=self.config.generators or None,
            max_per_category=self.config.max_per_category,
            seed=self.config.seed,
        )

        if not base_qs:
            raise RuntimeError("No valid questions generated from GMS store")

        dm = self.design_matrix.generate()
        doe_qs = self.rephraser.expand(base_qs, dm)

        print(f"\n  DOE expansion: {len(base_qs)} base x {len(dm)} runs "
              f"= {len(doe_qs)} questions")
        print(f"  Factors: {[f['name'] for f in self.factors]}")

        # Trace question generation
        if self.tracing:
            from collections import Counter
            cat_counts = dict(Counter(q.category for q in base_qs))
            self.tracing.trace_question_generation(
                run_id=self.config.resolve_store_name(),
                n_generators=len(set(q.category for q in base_qs)),
                n_base_questions=len(base_qs),
                n_doe_questions=len(doe_qs),
                category_counts=cat_counts,
                factors=[f["name"] for f in self.factors],
                n_runs=self.config.n_runs,
            )

        self._doe_questions = doe_qs
        return doe_qs

    # ----- Step 3+4: Run & Score -----------------------------------------

    def run(
        self,
        evaluator: Callable[[str, str], str] | None = None,
        llm_client=None,
        model: str = "",
    ) -> DOEBenchmarkResult:
        """Run the full benchmark: evaluate every DOE question and score.

        Provide one of:
        - ``evaluator``: callable(question, context) -> answer
        - ``llm_client``: Anthropic client (uses ``model`` for direct calls)
        """
        if evaluator is None and llm_client is None:
            raise ValueError("Provide evaluator or llm_client")

        if self._doe_questions is None:
            self.generate_questions()
        doe_questions = self._doe_questions

        model = model or self.config.target_model
        context_mode = self.config.context_mode
        section_map = _build_section_map(self.markdown)
        ctx_rng = np.random.default_rng(self.config.seed)

        model_label = model if llm_client else repr(evaluator)
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        n_base = len(set(dq.base_qid for dq in doe_questions))

        result = DOEBenchmarkResult(
            instance_name=self.config.resolve_store_name(),
            total_questions=len(doe_questions),
            n_base_questions=n_base,
            n_design_runs=self.config.n_runs,
            factors=[f["name"] for f in self.factors],
            model=model_label,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            graph_stats=self.store.stats() if self.store else {},
        )
        result.context_mode = context_mode

        # Trace run start
        if self.tracing:
            self.tracing.trace_run_start(
                run_id=run_id,
                markdown_path=self.config.markdown_path,
                target_model=model_label,
                factors=[f["name"] for f in self.factors],
                n_runs=self.config.n_runs,
                n_base_questions=n_base,
                context_mode=context_mode,
            )

        # GMSH Phase 4: Pre-run snapshot
        if self.snapshots and self.store:
            self.snapshots.take_snapshot(
                self.store, run_id=run_id, trigger="pre_run")

        print(f"\n  Evaluating {len(doe_questions)} DOE questions "
              f"(context_mode={context_mode})...")

        for i, dq in enumerate(doe_questions):
            q = dq.question
            tag = f"[{i + 1}/{len(doe_questions)}] {dq.doe_qid}"
            t0 = time.time()

            try:
                print(f"  {tag}...", end=" ", flush=True)

                ctx = _get_context(
                    dq, self.markdown, section_map, context_mode, ctx_rng)

                if evaluator is not None:
                    raw = evaluator(q.llm_prompt, ctx)
                else:
                    from knowlytix.benchmark.eval.llm_caller import call_llm
                    raw = call_llm(llm_client, ctx, q.llm_prompt, model)

                parser = PARSERS.get(q.answer_type, PARSERS["str"])
                answer = parser(raw)
                score = score_answer(answer, q.ground_truth)
                elapsed = time.time() - t0

                # GMS geometric judging (beyond binary correct/incorrect)
                verdict: GeometricVerdict | None = None
                if self.judge:
                    verdict = self.judge.judge(
                        llm_answer=raw,
                        ground_truth=q.ground_truth,
                        exact_correct=score.correct,
                        exact_detail=score.detail,
                        question_metadata=q.metadata,
                    )

                # GMSH typed verification (claim-level)
                typed_verdict = None
                if (self.config.enable_typed_verdicts
                        and self.judge
                        and hasattr(self.judge, "judge_typed")):
                    typed_verdict = self.judge.judge_typed(
                        llm_answer=raw,
                        ground_truth=q.ground_truth,
                        exact_correct=score.correct,
                        exact_detail=score.detail,
                        question_metadata=q.metadata,
                        question_category=q.category,
                        run_id=run_id,
                        question_id=dq.doe_qid,
                        question_text=q.natural_language,
                    )
                    self._typed_verdicts[dq.doe_qid] = typed_verdict

                if score.correct:
                    result.total_correct += 1

                detail = (verdict.detail if verdict
                          else score.detail)

                qr = DOEQuestionResult(
                    doe_qid=dq.doe_qid,
                    base_qid=dq.base_qid,
                    category=q.category,
                    question=q.natural_language,
                    ground_truth=str(q.ground_truth),
                    answer=str(answer),
                    correct=score.correct,
                    detail=detail,
                    factor_assignment=dq.factor_assignment,
                    raw_response=raw[:500] if raw else "",
                )

                # Inject geometric scores into factor_assignment for analysis
                if verdict:
                    qr.factor_assignment = {
                        **dq.factor_assignment,
                        "_geo_confidence": round(
                            verdict.geodesic_confidence, 4),
                        "_tension": round(verdict.tension_score, 4),
                        "_holonomy": round(verdict.holonomy_score, 4),
                        "_overall_confidence": round(
                            verdict.overall_confidence, 4),
                    }

                status = "PASS" if score.correct else "FAIL"
                geo_str = (f" geo={verdict.overall_confidence:.2f}"
                           if verdict else "")
                print(f"{status}{geo_str} ({elapsed:.1f}s)")

                # Trace (includes geometric scores)
                if self.tracing:
                    trace_assignment = dict(dq.factor_assignment)
                    if verdict:
                        trace_assignment["_geo_confidence"] = str(
                            round(verdict.geodesic_confidence, 4))
                        trace_assignment["_tension"] = str(
                            round(verdict.tension_score, 4))
                        trace_assignment["_holonomy"] = str(
                            round(verdict.holonomy_score, 4))
                        trace_assignment["_overall_confidence"] = str(
                            round(verdict.overall_confidence, 4))

                    self.tracing.trace_question(
                        doe_qid=dq.doe_qid,
                        base_qid=dq.base_qid,
                        factor_assignment=trace_assignment,
                        target_model=model_label,
                        question_text=q.natural_language,
                        answer=str(answer),
                        ground_truth=str(q.ground_truth),
                        correct=score.correct,
                        detail=detail,
                        elapsed=elapsed,
                        run_id=run_id,
                    )

                    # Trace typed verdict claims and failures
                    if typed_verdict and hasattr(self.tracing, "trace_typed_verdict"):
                        self.tracing.trace_typed_verdict(
                            dq.doe_qid, typed_verdict, run_id)

            except Exception as e:
                elapsed = time.time() - t0
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

                if self.tracing:
                    self.tracing.trace_question(
                        doe_qid=dq.doe_qid,
                        base_qid=dq.base_qid,
                        factor_assignment=dq.factor_assignment,
                        target_model=model_label,
                        question_text=q.natural_language,
                        answer=f"ERROR: {e}",
                        ground_truth=str(q.ground_truth),
                        correct=False,
                        detail=f"error: {e}",
                        elapsed=elapsed,
                        run_id=run_id,
                    )

            result.questions.append(qr)

        # Trace run end
        if self.tracing:
            self.tracing.trace_run_end(
                run_id, result.total_questions, result.total_correct)

        print(f"\n  Result: {result.total_correct}/{result.total_questions} "
              f"({result.accuracy:.1%})")
        return result

    @property
    def typed_verdicts(self) -> dict:
        """Access typed verdicts from the most recent run (GMSH Phase 1)."""
        return self._typed_verdicts

    # ----- GMS Baseline ---------------------------------------------------

    def run_gms_baseline(self) -> DOEBenchmarkResult:
        """Validate ground truth by running GMS answer functions.

        Should score ~100%. If not, there is a ground truth issue.
        """
        if self._doe_questions is None:
            self.generate_questions()
        doe_questions = self._doe_questions

        run_id = f"baseline_{uuid.uuid4().hex[:8]}"
        n_base = len(set(dq.base_qid for dq in doe_questions))

        result = DOEBenchmarkResult(
            instance_name=self.config.resolve_store_name(),
            total_questions=len(doe_questions),
            n_base_questions=n_base,
            n_design_runs=self.config.n_runs,
            factors=[f["name"] for f in self.factors],
            model="gms_baseline",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            graph_stats=self.store.stats() if self.store else {},
        )

        # Trace baseline run start
        if self.tracing:
            self.tracing.trace_run_start(
                run_id=run_id,
                markdown_path=self.config.markdown_path,
                target_model="gms_baseline",
                factors=[f["name"] for f in self.factors],
                n_runs=self.config.n_runs,
                n_base_questions=n_base,
                context_mode="gms_direct",
            )

        print(f"\n  GMS baseline: {len(doe_questions)} questions...")
        for dq in doe_questions:
            q = dq.question
            t0 = time.time()
            try:
                gms_answer = q.graph_answer_fn(None)  # closure ignores graph arg
                score = score_answer(gms_answer, q.ground_truth)
            except Exception as e:
                score = type("S", (), {"correct": False, "detail": f"error: {e}"})()
                gms_answer = f"ERROR: {e}"
            elapsed = time.time() - t0

            if score.correct:
                result.total_correct += 1

            result.questions.append(DOEQuestionResult(
                doe_qid=dq.doe_qid,
                base_qid=dq.base_qid,
                category=q.category,
                question=q.natural_language,
                ground_truth=str(q.ground_truth),
                answer=str(gms_answer),
                correct=score.correct,
                detail=score.detail,
                factor_assignment=dq.factor_assignment,
            ))

            # Trace each baseline question
            if self.tracing:
                self.tracing.trace_question(
                    doe_qid=dq.doe_qid,
                    base_qid=dq.base_qid,
                    factor_assignment=dq.factor_assignment,
                    target_model="gms_baseline",
                    question_text=q.natural_language,
                    answer=str(gms_answer),
                    ground_truth=str(q.ground_truth),
                    correct=score.correct,
                    detail=score.detail,
                    elapsed=elapsed,
                    run_id=run_id,
                )

        # Trace baseline run end
        if self.tracing:
            self.tracing.trace_run_end(
                run_id, result.total_questions, result.total_correct)

        print(f"  GMS baseline: {result.total_correct}/{result.total_questions} "
              f"({result.accuracy:.0%})")
        if result.total_correct != result.total_questions:
            print("  WARNING: GMS baseline is not 100% — ground truth issue!")
        return result

    # ----- Scenario execution -----------------------------------------------

    def run_scenarios(
        self,
        scenarios: list,
        evaluator: Callable[[str, str], str] | None = None,
        llm_client=None,
        model: str = "",
    ) -> DOEBenchmarkResult:
        """Run authored scenarios through the DOE benchmark pipeline.

        Converts each ``Scenario`` into a ``FSBQuestion``, applies DOE
        expansion, and feeds into the existing ``run()`` pipeline.

        Parameters
        ----------
        scenarios : list[Scenario]
            Scenarios to evaluate.
        evaluator : callable or None
            ``(question, context) -> answer`` function.
        llm_client : object or None
            Anthropic client for direct calls.
        model : str
            Model name.

        Returns
        -------
        DOEBenchmarkResult
            Combined results from all scenarios.
        """
        if self.store is None:
            raise RuntimeError("Call ingest() first")

        from .bridge import scenario_to_fsb_question

        fsb_questions = []
        for scn in scenarios:
            q = scenario_to_fsb_question(scn, self.store)
            if q is not None:
                fsb_questions.append(q)

        if not fsb_questions:
            raise RuntimeError(
                f"No valid questions from {len(scenarios)} scenarios")

        print(f"\n  Scenarios: {len(fsb_questions)} questions "
              f"from {len(scenarios)} scenarios")

        # Expand via DOE design matrix
        dm = self.design_matrix.generate()
        doe_qs = self.rephraser.expand(fsb_questions, dm)

        print(f"  DOE expansion: {len(fsb_questions)} base x {len(dm)} runs "
              f"= {len(doe_qs)} questions")

        # Stash expanded questions and run via standard pipeline
        self._doe_questions = doe_qs
        return self.run(
            evaluator=evaluator,
            llm_client=llm_client,
            model=model,
        )

    # ----- Step 7: Analysis -----------------------------------------------

    def analyze(self, result: DOEBenchmarkResult) -> DOEAnalyzer:
        """Run statistical analysis on benchmark results."""
        analyzer = DOEAnalyzer(result)

        # Trace analysis results
        if self.tracing:
            analysis_dict = {}
            try:
                logistic = analyzer.run_logistic()
                for factor, stats in logistic.items():
                    analysis_dict[factor] = stats
            except Exception:
                pass
            if analysis_dict:
                # Extract run_id from first question's tracing
                run_id = "unknown"
                if result.questions:
                    run_id = f"analysis_{result.timestamp.replace(' ', '_')}"
                self.tracing.trace_analysis(run_id, analysis_dict)
                self.tracing.save()

        return analyzer

    # ----- Printing -------------------------------------------------------

    @staticmethod
    def print_results(result: DOEBenchmarkResult):
        """Print summary of DOE benchmark results."""
        n = result.total_questions
        print(f"\n{'=' * 70}")
        print(f"DOE-GMS BENCHMARK — {result.instance_name}")
        print(f"{'=' * 70}")
        print(f"  Model:          {result.model}")
        print(f"  Base questions:  {result.n_base_questions}")
        print(f"  Design runs:    {result.n_design_runs}")
        print(f"  Total questions: {n}")
        print(f"  Factors:         {result.factors}")
        print(f"  Accuracy:        {result.total_correct}/{n} "
              f"({result.accuracy:.1%})")

        df = result.to_dataframe()
        if not df.empty:
            has_geo = "_overall_confidence" in df.columns
            header = f"  {'Category':<22} {'Correct':>7} {'Total':>6} {'Rate':>6}"
            if has_geo:
                header += f" {'GeoConf':>8} {'Tension':>8}"
            print(f"\n{header}")
            print(f"  {'─' * (55 + (18 if has_geo else 0))}")
            for cat in sorted(df["category"].unique()):
                sub = df[df["category"] == cat]
                correct = sub["correct"].sum()
                total = len(sub)
                line = (f"  {cat:<22} {correct:>7} {total:>6} "
                        f"{correct / total:>5.0%}")
                if has_geo:
                    geo = sub["_overall_confidence"].mean()
                    ten = sub["_tension"].mean()
                    line += f"  {geo:>7.3f}  {ten:>7.3f}"
                print(line)

            # Geometric summary
            if has_geo:
                print("\n  Geometric Summary:")
                print(f"    Mean geodesic confidence: "
                      f"{df['_geo_confidence'].mean():.3f}")
                print(f"    Mean tension score:       "
                      f"{df['_tension'].mean():.3f}")
                print(f"    Mean holonomy score:      "
                      f"{df['_holonomy'].mean():.3f}")
                print(f"    Mean overall confidence:  "
                      f"{df['_overall_confidence'].mean():.3f}")

    # ----- Report ------------------------------------------------------------

    def generate_report(
        self,
        result: DOEBenchmarkResult,
        analyzer: DOEAnalyzer | None = None,
        output_path: str = "doe_report.html",
        extra_results: list[DOEBenchmarkResult] | None = None,
    ) -> str:
        """Generate an HTML dashboard report.

        Includes benchmark results, factor analysis, geometric scores,
        and tracing agent summary.
        """
        from .dashboard import generate_report
        path = generate_report(
            result,
            analyzer=analyzer,
            output_path=output_path,
            extra_results=extra_results,
            tracing=self.tracing,
        )

        # Trace report generation
        if self.tracing:
            run_id = result.timestamp.replace(" ", "_") if result.timestamp else "unknown"
            self.tracing.trace_report(
                run_id=run_id,
                report_type="dashboard",
                output_path=output_path,
            )

        return path

    # ----- GMSH Audit Report ------------------------------------------------

    def generate_audit_report(
        self,
        result: DOEBenchmarkResult,
        analyzer: DOEAnalyzer | None = None,
        output_path: str = "doe_audit_report.html",
        accuracy_threshold: float = 0.80,
        geo_confidence_threshold: float = 0.50,
        failure_budget: dict[str, int] | None = None,
    ) -> str:
        """Generate an enhanced GMSH audit report with release gate.

        Requires ``enable_typed_verdicts=True`` in config.

        Parameters
        ----------
        result : DOEBenchmarkResult
            Benchmark results.
        analyzer : DOEAnalyzer or None
            Statistical analysis results.
        output_path : str
            Where to write the HTML report.
        accuracy_threshold : float
            Minimum pass rate for release gate.
        geo_confidence_threshold : float
            Minimum geometric confidence for release gate.
        failure_budget : dict[str, int] or None
            Max allowed failures per code.

        Returns
        -------
        str
            The output path.
        """
        if not self._typed_verdicts:
            print("  WARNING: No typed verdicts — falling back to base report")
            return self.generate_report(result, analyzer, output_path)

        from .audit import AuditReporter
        from .dashboard import generate_enhanced_report

        reporter = AuditReporter(
            result, self._typed_verdicts, self.truth_store)

        gate = reporter.generate_release_gate(
            accuracy_threshold=accuracy_threshold,
            geo_confidence_threshold=geo_confidence_threshold,
            failure_budget=failure_budget or {},
        )

        # Trace release gate evaluation
        if self.tracing:
            run_id = result.timestamp.replace(" ", "_") if result.timestamp else "unknown"
            self.tracing.trace_release_gate(
                run_id=run_id,
                passed=gate.passed,
                accuracy_threshold=gate.accuracy_threshold,
                actual_accuracy=gate.actual_accuracy,
                geo_threshold=gate.geo_confidence_threshold,
                actual_geo=gate.actual_geo_confidence,
                blocking_failures=gate.blocking_failures,
                failure_counts=gate.actual_failures,
            )

        # Print executive summary to console
        summary = reporter.generate_executive_summary()
        print(f"\n{summary}\n")

        path = generate_enhanced_report(
            result,
            typed_verdicts=self._typed_verdicts,
            gate=gate,
            analyzer=analyzer,
            output_path=output_path,
            tracing=self.tracing,
        )

        # Trace audit report generation
        if self.tracing:
            run_id = result.timestamp.replace(" ", "_") if result.timestamp else "unknown"
            self.tracing.trace_report(
                run_id=run_id,
                report_type="audit",
                output_path=output_path,
            )

        return path

    # ----- Persistence ----------------------------------------------------

    def save_tracing(self):
        """Persist the tracing GMS store."""
        if self.tracing:
            self.tracing.save()
            stats = self.tracing.stats()
            print(f"  Tracing saved: {stats['entities']} entities, "
                  f"{stats['triples']} triples")
