# SPDX-License-Identifier: Apache-2.0
"""DOE Tracing GMS — records every benchmark event as geometric triples.

Follows the TracingGMS pattern used internally by the Knowlytix DS-agent
(private monorepo) but with DOE-specific seed triples and helper methods
for recording factor assignments, scores, and statistical analysis results.
"""

from __future__ import annotations

import os
import time
from typing import Any

import torch

from knowlytix.doc.config import DocGMSConfig
from knowlytix.doc.extract import ExtractedTriple
from knowlytix.doc.learn import RuntimeLearner
from knowlytix.doc.store import GMSExpertStore


class DOETracingGMS:
    """Tracing store that captures the full DOE benchmark execution trace.

    Records:
    - ``run_<id>``: benchmark-level metadata (document, target model, factors)
    - ``doe_q_<id>``: per-question trace (factor assignment, answer, score)
    - ``analysis_<id>``: statistical analysis results (significant factors)
    """

    SEED_TRIPLES = [
        ("doe_benchmark", "is_type", "evaluation"),
        ("doe_question", "has_property", "factor_assignment"),
        ("doe_question", "has_property", "score"),
        ("doe_question", "has_property", "ground_truth"),
        ("doe_question", "has_property", "answer"),
        ("doe_run", "is_part_of", "doe_benchmark"),
        ("doe_analysis", "is_type", "statistical_analysis"),
        # Factor category seeds
        ("query_factor", "is_type", "doe_factor"),
        ("persona_factor", "is_type", "doe_factor"),
        ("reasoning_factor", "is_type", "doe_factor"),
        ("context_factor", "is_type", "doe_factor"),
        ("entity_factor", "is_type", "doe_factor"),
        ("adversarial_factor", "is_type", "doe_factor"),
        ("variability_factor", "is_type", "doe_factor"),
        ("robustness_factor", "is_type", "doe_factor"),
        # Evaluation concepts
        ("correct", "is_type", "score_outcome"),
        ("incorrect", "is_type", "score_outcome"),
        ("logistic_regression", "is_type", "analysis_method"),
        ("odds_ratio", "is_type", "effect_size"),
        # GMSH pipeline event seeds
        ("doe_ingest", "is_type", "pipeline_event"),
        ("doe_question_generation", "is_type", "pipeline_event"),
        ("doe_baseline_run", "is_type", "doe_run"),
        ("gms_snapshot", "is_type", "versioning_event"),
        ("truth_store_build", "is_type", "pipeline_event"),
        ("semantic_judge_call", "is_type", "verification_event"),
        ("release_gate", "is_type", "quality_gate"),
        ("doe_report", "is_type", "pipeline_event"),
        # Agentic harness seeds (WS1-WS7)
        ("gateway_decision", "is_type", "interception_event"),
        ("fault_injection", "is_type", "gateway_event"),
        ("policy_decision", "is_type", "policy_event"),
        ("policy_rule", "is_type", "policy_definition"),
        ("test_scenario", "is_type", "scenario"),
        ("scenario_registry", "is_type", "registry"),
        ("run_state", "is_type", "state_machine_event"),
        ("state_transition", "is_type", "lifecycle_event"),
        ("replay_manifest", "is_type", "replay_artifact"),
        ("regression_suite", "is_type", "test_collection"),
        ("run_comparison", "is_type", "analysis_event"),
        ("process_evaluation", "is_type", "evaluation_event"),
        ("recovery_evaluation", "is_type", "evaluation_event"),
        ("interaction_evaluation", "is_type", "evaluation_event"),
        ("risk_tier_gate", "is_type", "quality_gate"),
        ("run_trend", "is_type", "trend_analysis"),
        # Black-box testing enhancement seeds
        ("target_profile", "is_type", "sut_definition"),
        ("test_campaign", "is_type", "campaign"),
        ("campaign_run", "is_part_of", "test_campaign"),
        ("environment_simulator", "is_type", "test_infrastructure"),
        ("environment_state_change", "is_type", "side_effect"),
        ("oracle_verdict", "is_type", "evaluation_event"),
        ("severity_assessment", "is_type", "classification"),
        ("severity_critical", "is_type", "severity_level"),
        ("severity_high", "is_type", "severity_level"),
        ("severity_moderate", "is_type", "severity_level"),
        ("severity_low", "is_type", "severity_level"),
        # GMS-native enhancement seeds (Phase 1: profiling + cost)
        ("performance_profile", "is_type", "profiling_event"),
        ("sla_compliance", "is_type", "profiling_metric"),
        ("cost_record", "is_type", "cost_event"),
        ("budget_compliance", "is_type", "cost_metric"),
        ("mock_injection", "is_type", "gateway_event"),
        ("stale_data_injection", "is_type", "gateway_event"),
        # GMS-native enhancement seeds (Phase 2: memory, hallucination, calibration)
        ("memory_probe", "is_type", "stateful_test"),
        ("memory_recall", "is_type", "evaluation_event"),
        ("memory_degradation", "is_type", "drift_metric"),
        ("hallucination_assessment", "is_type", "evaluation_event"),
        ("hallucination_claim", "is_type", "claim_analysis"),
        ("calibration_point", "is_type", "calibration_event"),
        ("calibration_report", "is_type", "calibration_metric"),
        # GMS-native enhancement seeds (Phase 3: multi-agent, cross-doc, disambiguation)
        ("multi_agent_interaction", "is_type", "multi_agent_test"),
        ("inter_agent_tension", "is_type", "consistency_metric"),
        ("delegation_holonomy", "is_type", "process_metric"),
        ("cross_document_conflict", "is_type", "conflict_detection"),
        ("disambiguation_test", "is_type", "disambiguation_event"),
        # GMS-native enhancement seeds (Phase 4: invariance, streaming)
        ("paraphrase_invariance", "is_type", "invariance_test"),
        ("stream_consistency", "is_type", "streaming_test"),
        ("stream_chunk", "is_type", "streaming_event"),
        # GMS-native enhancement seeds (Phase 5: dashboard)
        ("dashboard_snapshot", "is_type", "monitoring_event"),
    ]

    def __init__(self, stores_dir: str, device: torch.device):
        self.device = device
        store_path = os.path.join(stores_dir, "_doe_tracing")
        os.makedirs(store_path, exist_ok=True)

        config = DocGMSConfig(store_path=store_path)
        self.config = config
        self.store = GMSExpertStore(config, device)
        self.learner: RuntimeLearner | None = None

        if self.store.exists():
            self.store.load()
            self.learner = RuntimeLearner(self.store, config.learn)
        else:
            self._bootstrap()

    def _bootstrap(self):
        from knowlytix.bench.graph import DocumentGraph
        from knowlytix.core.train_finstructbench import GraphToGMS, train_gms, populate_enm
        from knowlytix.core.graph.transport import RelationalTransport

        doc_graph = DocumentGraph()
        for h, r, t in self.SEED_TRIPLES:
            doc_graph.add_triple(h, r, t)

        self.store.doc_graph = doc_graph
        adapter = GraphToGMS(doc_graph)
        model = train_gms(
            adapter, self.device,
            epochs=self.config.train.epochs,
            batch_size=self.config.train.batch_size,
            geometry=self.config.geometry,
        )
        enm = populate_enm(doc_graph, adapter)
        transport = RelationalTransport(model)

        self.store.model = model
        self.store.adapter = adapter
        self.store.enm = enm
        self.store.transport = transport
        self.store.markdown = "DOE benchmark tracing GMS — full execution audit log."
        self.store.save()
        self.learner = RuntimeLearner(self.store, self.config.learn)

    # -- Learning helpers ---------------------------------------------------

    def _learn(self, triples: list[ExtractedTriple]):
        if not self.learner or not triples:
            return None
        try:
            result = self.learner.learn_triples(triples, self.device)
            if result and result.training_queued:
                self.learner.flush_training()
                self.learner.wait_for_training(timeout=30.0)
            return result
        except Exception as e:
            print(f"  [doe-tracing] Learn error: {e}")
            return None

    # -- High-level tracing API --------------------------------------------

    def trace_run_start(
        self,
        run_id: str,
        markdown_path: str,
        target_model: str,
        factors: list[str],
        n_runs: int,
        n_base_questions: int,
        context_mode: str,
    ):
        """Record the start of a benchmark run."""
        triples = [
            ExtractedTriple(head=run_id, relation="is_type", tail="doe_run"),
            ExtractedTriple(head=run_id, relation="document",
                            tail=os.path.basename(markdown_path)),
            ExtractedTriple(head=run_id, relation="target_model",
                            tail=target_model),
            ExtractedTriple(head=run_id, relation="context_mode",
                            tail=context_mode),
            ExtractedTriple(head=run_id, relation="n_design_runs",
                            tail=str(n_runs),
                            numeric_value=float(n_runs)),
            ExtractedTriple(head=run_id, relation="n_base_questions",
                            tail=str(n_base_questions),
                            numeric_value=float(n_base_questions)),
            ExtractedTriple(head=run_id, relation="started_at",
                            tail=time.strftime("%Y-%m-%d %H:%M:%S")),
        ]
        for f in factors:
            triples.append(ExtractedTriple(
                head=run_id, relation="tests_factor", tail=f))
        self._learn(triples)

    def trace_question(
        self,
        doe_qid: str,
        base_qid: str,
        factor_assignment: dict[str, Any],
        target_model: str,
        question_text: str,
        answer: str,
        ground_truth: str,
        correct: bool,
        detail: str,
        elapsed: float,
        run_id: str,
    ):
        """Record a single DOE question evaluation."""
        triples = [
            ExtractedTriple(head=doe_qid, relation="is_type",
                            tail="doe_question"),
            ExtractedTriple(head=doe_qid, relation="base_question",
                            tail=base_qid),
            ExtractedTriple(head=doe_qid, relation="target_model",
                            tail=target_model),
            ExtractedTriple(head=doe_qid, relation="score_correct",
                            tail=str(correct).lower()),
            ExtractedTriple(head=doe_qid, relation="score_detail",
                            tail=detail[:200]),
            ExtractedTriple(head=doe_qid, relation="elapsed_seconds",
                            tail=str(round(elapsed, 3)),
                            numeric_value=round(elapsed, 3)),
            ExtractedTriple(head=doe_qid, relation="run",
                            tail=run_id),
            ExtractedTriple(head=doe_qid, relation="timestamp",
                            tail=time.strftime("%Y-%m-%d %H:%M:%S")),
        ]
        # Factor assignments as individual triples
        for factor_name, level in factor_assignment.items():
            triples.append(ExtractedTriple(
                head=doe_qid,
                relation=f"factor_{factor_name}",
                tail=str(level),
            ))
        self._learn(triples)

    def trace_run_end(self, run_id: str, total: int, correct: int):
        """Record the end of a benchmark run."""
        accuracy = correct / total if total > 0 else 0.0
        triples = [
            ExtractedTriple(head=run_id, relation="total_questions",
                            tail=str(total), numeric_value=float(total)),
            ExtractedTriple(head=run_id, relation="total_correct",
                            tail=str(correct), numeric_value=float(correct)),
            ExtractedTriple(head=run_id, relation="accuracy",
                            tail=f"{accuracy:.4f}", numeric_value=accuracy),
            ExtractedTriple(head=run_id, relation="finished_at",
                            tail=time.strftime("%Y-%m-%d %H:%M:%S")),
        ]
        self._learn(triples)

    def trace_analysis(self, run_id: str, analysis_results: dict):
        """Record DOE statistical analysis results.

        Parameters
        ----------
        analysis_results : dict
            Keys: factor names, values: dicts with 'p_value', 'deviance',
            'significant', 'odds_ratios' (from DOEAnalyzer.run_logistic).
        """
        analysis_id = f"analysis_{run_id}"
        triples = [
            ExtractedTriple(head=analysis_id, relation="is_type",
                            tail="doe_analysis"),
            ExtractedTriple(head=analysis_id, relation="run", tail=run_id),
            ExtractedTriple(head=analysis_id, relation="timestamp",
                            tail=time.strftime("%Y-%m-%d %H:%M:%S")),
        ]
        for factor, stats in analysis_results.items():
            p_val = stats.get("p_value", 1.0)
            deviance = stats.get("deviance", 0.0)
            significant = stats.get("significant", False)
            triples.append(ExtractedTriple(
                head=analysis_id,
                relation=f"factor_{factor}_p_value",
                tail=f"{p_val:.6f}",
                numeric_value=p_val,
            ))
            triples.append(ExtractedTriple(
                head=analysis_id,
                relation=f"factor_{factor}_deviance",
                tail=f"{deviance:.4f}",
                numeric_value=deviance,
            ))
            if significant:
                triples.append(ExtractedTriple(
                    head=analysis_id,
                    relation="significant_factor",
                    tail=factor,
                ))
        self._learn(triples)

    # -- GMSH pipeline tracing (new) ----------------------------------------

    def trace_ingest(
        self,
        store_name: str,
        markdown_path: str,
        loaded_existing: bool,
        elapsed: float,
        store_stats: dict,
    ):
        """Record a document ingestion event.

        Parameters
        ----------
        store_name : str
            Name of the GMS store.
        markdown_path : str
            Path to the source markdown.
        loaded_existing : bool
            Whether an existing store was loaded (vs fresh ingest).
        elapsed : float
            Ingestion time in seconds.
        store_stats : dict
            Store statistics after ingest (entities, triples, enm_entries).
        """
        ingest_id = f"ingest_{store_name}"
        triples = [
            ExtractedTriple(head=ingest_id, relation="is_type",
                            tail="doe_ingest"),
            ExtractedTriple(head=ingest_id, relation="store_name",
                            tail=store_name),
            ExtractedTriple(head=ingest_id, relation="document",
                            tail=os.path.basename(markdown_path)),
            ExtractedTriple(head=ingest_id, relation="loaded_existing",
                            tail=str(loaded_existing).lower()),
            ExtractedTriple(head=ingest_id, relation="elapsed_seconds",
                            tail=str(round(elapsed, 3)),
                            numeric_value=round(elapsed, 3)),
            ExtractedTriple(head=ingest_id, relation="timestamp",
                            tail=time.strftime("%Y-%m-%d %H:%M:%S")),
        ]
        for key in ("entities", "triples", "enm_entries"):
            val = store_stats.get(key)
            if val is not None:
                triples.append(ExtractedTriple(
                    head=ingest_id,
                    relation=f"stat_{key}",
                    tail=str(val),
                    numeric_value=float(val),
                ))
        self._learn(triples)

    def trace_question_generation(
        self,
        run_id: str,
        n_generators: int,
        n_base_questions: int,
        n_doe_questions: int,
        category_counts: dict[str, int],
        factors: list[str],
        n_runs: int,
    ):
        """Record question generation and DOE expansion.

        Parameters
        ----------
        run_id : str
            Identifier for this generation pass.
        n_generators : int
            Number of generators that ran.
        n_base_questions : int
            Total base questions before DOE expansion.
        n_doe_questions : int
            Total questions after DOE expansion.
        category_counts : dict[str, int]
            Base question count per generator category.
        factors : list[str]
            DOE factor names used for expansion.
        n_runs : int
            Number of DOE design runs.
        """
        gen_id = f"qgen_{run_id}"
        triples = [
            ExtractedTriple(head=gen_id, relation="is_type",
                            tail="doe_question_generation"),
            ExtractedTriple(head=gen_id, relation="n_generators",
                            tail=str(n_generators),
                            numeric_value=float(n_generators)),
            ExtractedTriple(head=gen_id, relation="n_base_questions",
                            tail=str(n_base_questions),
                            numeric_value=float(n_base_questions)),
            ExtractedTriple(head=gen_id, relation="n_doe_questions",
                            tail=str(n_doe_questions),
                            numeric_value=float(n_doe_questions)),
            ExtractedTriple(head=gen_id, relation="n_design_runs",
                            tail=str(n_runs),
                            numeric_value=float(n_runs)),
            ExtractedTriple(head=gen_id, relation="timestamp",
                            tail=time.strftime("%Y-%m-%d %H:%M:%S")),
        ]
        for cat, count in category_counts.items():
            triples.append(ExtractedTriple(
                head=gen_id,
                relation=f"category_{cat}",
                tail=str(count),
                numeric_value=float(count),
            ))
        for f in factors:
            triples.append(ExtractedTriple(
                head=gen_id, relation="uses_factor", tail=f))
        self._learn(triples)

    def trace_semantic_verdict(
        self,
        doe_qid: str,
        run_id: str,
        judge_model: str,
        rubric_scores: dict[str, int],
        weighted_score: float,
        entity_coverage: float,
        n_geodesic_pairs: int,
        n_tension_pairs: int,
        calibrated_confidence: float,
        passed: bool,
        reasoning: str = "",
    ):
        """Record an LLM-as-judge semantic verification call.

        Parameters
        ----------
        doe_qid : str
            DOE question identifier.
        run_id : str
            Benchmark run identifier.
        judge_model : str
            Model used for the LLM judge.
        rubric_scores : dict[str, int]
            Per-dimension rubric scores (1-5).
        weighted_score : float
            Weighted rubric score.
        entity_coverage : float
            Fraction of answer entities found in GMS.
        n_geodesic_pairs : int
            Number of geodesic proximity pairs computed.
        n_tension_pairs : int
            Number of tension energy pairs computed.
        calibrated_confidence : float
            Final GMS-calibrated confidence.
        passed : bool
            Whether the semantic claim passed.
        reasoning : str
            LLM judge reasoning text.
        """
        sem_id = f"sem_{doe_qid}"
        triples = [
            ExtractedTriple(head=sem_id, relation="is_type",
                            tail="semantic_judge_call"),
            ExtractedTriple(head=sem_id, relation="question",
                            tail=doe_qid),
            ExtractedTriple(head=sem_id, relation="run",
                            tail=run_id),
            ExtractedTriple(head=sem_id, relation="judge_model",
                            tail=judge_model),
            ExtractedTriple(head=sem_id, relation="weighted_score",
                            tail=f"{weighted_score:.3f}",
                            numeric_value=weighted_score),
            ExtractedTriple(head=sem_id, relation="entity_coverage",
                            tail=f"{entity_coverage:.3f}",
                            numeric_value=entity_coverage),
            ExtractedTriple(head=sem_id, relation="n_geodesic_pairs",
                            tail=str(n_geodesic_pairs),
                            numeric_value=float(n_geodesic_pairs)),
            ExtractedTriple(head=sem_id, relation="n_tension_pairs",
                            tail=str(n_tension_pairs),
                            numeric_value=float(n_tension_pairs)),
            ExtractedTriple(head=sem_id, relation="calibrated_confidence",
                            tail=f"{calibrated_confidence:.4f}",
                            numeric_value=calibrated_confidence),
            ExtractedTriple(head=sem_id, relation="passed",
                            tail=str(passed).lower()),
            ExtractedTriple(head=sem_id, relation="timestamp",
                            tail=time.strftime("%Y-%m-%d %H:%M:%S")),
        ]
        for dim, score in rubric_scores.items():
            triples.append(ExtractedTriple(
                head=sem_id,
                relation=f"rubric_{dim}",
                tail=str(score),
                numeric_value=float(score),
            ))
        if reasoning:
            triples.append(ExtractedTriple(
                head=sem_id,
                relation="reasoning",
                tail=reasoning[:200],
            ))
        self._learn(triples)

    def trace_truth_store(
        self,
        store_name: str,
        elapsed: float,
        truth_counts: dict[str, int],
        total: int,
    ):
        """Record truth store build event.

        Parameters
        ----------
        store_name : str
            GMS store name the truths were built from.
        elapsed : float
            Build time in seconds.
        truth_counts : dict[str, int]
            Count of truth objects by type (entity, relation, threshold, artifact).
        total : int
            Total truth objects created.
        """
        ts_id = f"truth_store_{store_name}"
        triples = [
            ExtractedTriple(head=ts_id, relation="is_type",
                            tail="truth_store_build"),
            ExtractedTriple(head=ts_id, relation="store_name",
                            tail=store_name),
            ExtractedTriple(head=ts_id, relation="total_truths",
                            tail=str(total),
                            numeric_value=float(total)),
            ExtractedTriple(head=ts_id, relation="elapsed_seconds",
                            tail=str(round(elapsed, 3)),
                            numeric_value=round(elapsed, 3)),
            ExtractedTriple(head=ts_id, relation="timestamp",
                            tail=time.strftime("%Y-%m-%d %H:%M:%S")),
        ]
        for truth_type, count in truth_counts.items():
            triples.append(ExtractedTriple(
                head=ts_id,
                relation=f"truth_{truth_type}_count",
                tail=str(count),
                numeric_value=float(count),
            ))
        self._learn(triples)

    def trace_release_gate(
        self,
        run_id: str,
        passed: bool,
        accuracy_threshold: float,
        actual_accuracy: float,
        geo_threshold: float,
        actual_geo: float,
        blocking_failures: list[str],
        failure_counts: dict[str, int],
    ):
        """Record a release gate evaluation.

        Parameters
        ----------
        run_id : str
            Benchmark run that was evaluated.
        passed : bool
            Whether the release gate passed.
        accuracy_threshold : float
            Required minimum accuracy.
        actual_accuracy : float
            Observed accuracy.
        geo_threshold : float
            Required minimum geometric confidence.
        actual_geo : float
            Observed geometric confidence.
        blocking_failures : list[str]
            Reasons the gate was blocked (empty if passed).
        failure_counts : dict[str, int]
            Failure code counts.
        """
        gate_id = f"gate_{run_id}"
        triples = [
            ExtractedTriple(head=gate_id, relation="is_type",
                            tail="release_gate"),
            ExtractedTriple(head=gate_id, relation="run",
                            tail=run_id),
            ExtractedTriple(head=gate_id, relation="passed",
                            tail=str(passed).lower()),
            ExtractedTriple(head=gate_id, relation="accuracy_threshold",
                            tail=f"{accuracy_threshold:.4f}",
                            numeric_value=accuracy_threshold),
            ExtractedTriple(head=gate_id, relation="actual_accuracy",
                            tail=f"{actual_accuracy:.4f}",
                            numeric_value=actual_accuracy),
            ExtractedTriple(head=gate_id, relation="geo_threshold",
                            tail=f"{geo_threshold:.4f}",
                            numeric_value=geo_threshold),
            ExtractedTriple(head=gate_id, relation="actual_geo_confidence",
                            tail=f"{actual_geo:.4f}",
                            numeric_value=actual_geo),
            ExtractedTriple(head=gate_id, relation="timestamp",
                            tail=time.strftime("%Y-%m-%d %H:%M:%S")),
        ]
        for reason in blocking_failures:
            triples.append(ExtractedTriple(
                head=gate_id,
                relation="blocked_by",
                tail=reason[:100],
            ))
        for code, count in failure_counts.items():
            triples.append(ExtractedTriple(
                head=gate_id,
                relation=f"failure_{code}",
                tail=str(count),
                numeric_value=float(count),
            ))
        self._learn(triples)

    def trace_report(
        self,
        run_id: str,
        report_type: str,
        output_path: str,
    ):
        """Record a report generation event.

        Parameters
        ----------
        run_id : str
            Benchmark run the report covers.
        report_type : str
            Type of report ("dashboard", "audit", "comparison").
        output_path : str
            File path where the report was written.
        """
        report_id = f"report_{run_id}_{report_type}"
        triples = [
            ExtractedTriple(head=report_id, relation="is_type",
                            tail="doe_report"),
            ExtractedTriple(head=report_id, relation="run",
                            tail=run_id),
            ExtractedTriple(head=report_id, relation="report_type",
                            tail=report_type),
            ExtractedTriple(head=report_id, relation="output_path",
                            tail=os.path.basename(output_path)),
            ExtractedTriple(head=report_id, relation="timestamp",
                            tail=time.strftime("%Y-%m-%d %H:%M:%S")),
        ]
        self._learn(triples)

    # -- GMSH typed verdict tracing ------------------------------------------

    def trace_typed_verdict(self, doe_qid: str, typed_verdict, run_id: str):
        """Record GMSH typed verdict claims and failures as GMS triples.

        Parameters
        ----------
        doe_qid : str
            DOE question identifier.
        typed_verdict : TypedVerdict
            The typed verdict from GMSJudge.judge_typed().
        run_id : str
            Benchmark run identifier.
        """
        triples = []

        # Record each claim
        for claim in getattr(typed_verdict, "claims", []):
            triples.append(ExtractedTriple(
                head=doe_qid,
                relation="has_claim",
                tail=claim.claim_id,
            ))
            triples.append(ExtractedTriple(
                head=claim.claim_id,
                relation="claim_type",
                tail=claim.claim_type,
            ))

        # Record each sub-verdict
        for sv in getattr(typed_verdict, "sub_verdicts", []):
            triples.append(ExtractedTriple(
                head=sv.claim_id,
                relation="verified_by",
                tail=sv.verifier_name,
            ))
            triples.append(ExtractedTriple(
                head=sv.claim_id,
                relation="verdict_passed",
                tail=str(sv.passed).lower(),
            ))
            if sv.gms_method_used:
                triples.append(ExtractedTriple(
                    head=sv.claim_id,
                    relation="gms_method",
                    tail=sv.gms_method_used,
                ))

        # Record failures
        for failure in getattr(typed_verdict, "failures", []):
            triples.append(ExtractedTriple(
                head=doe_qid,
                relation="failure_code",
                tail=failure.code.value if hasattr(failure.code, "value") else str(failure.code),
            ))
            triples.append(ExtractedTriple(
                head=failure.claim_id,
                relation="failed_with",
                tail=failure.code.value if hasattr(failure.code, "value") else str(failure.code),
            ))

        # Record summary counts
        n_claims = getattr(typed_verdict, "n_claims", 0)
        n_passed = getattr(typed_verdict, "n_passed", 0)
        n_failed = getattr(typed_verdict, "n_failed", 0)
        if n_claims > 0:
            triples.append(ExtractedTriple(
                head=doe_qid,
                relation="n_claims",
                tail=str(n_claims),
                numeric_value=float(n_claims),
            ))
            triples.append(ExtractedTriple(
                head=doe_qid,
                relation="n_claims_passed",
                tail=str(n_passed),
                numeric_value=float(n_passed),
            ))
            triples.append(ExtractedTriple(
                head=doe_qid,
                relation="n_claims_failed",
                tail=str(n_failed),
                numeric_value=float(n_failed),
            ))

        if triples:
            self._learn(triples)

    def trace_snapshot(self, snapshot_id: str, run_id: str, store_stats: dict):
        """Record a GMS store snapshot event as triples.

        Parameters
        ----------
        snapshot_id : str
            Unique snapshot identifier.
        run_id : str
            Benchmark run that triggered the snapshot.
        store_stats : dict
            Store statistics at snapshot time (entities, triples, enm_entries).
        """
        triples = [
            ExtractedTriple(
                head=snapshot_id, relation="is_type",
                tail="gms_snapshot"),
            ExtractedTriple(
                head=snapshot_id, relation="run",
                tail=run_id),
            ExtractedTriple(
                head=snapshot_id, relation="timestamp",
                tail=time.strftime("%Y-%m-%d %H:%M:%S")),
        ]
        for key in ("entities", "triples", "enm_entries"):
            val = store_stats.get(key)
            if val is not None:
                triples.append(ExtractedTriple(
                    head=snapshot_id,
                    relation=f"stat_{key}",
                    tail=str(val),
                    numeric_value=float(val),
                ))
        self._learn(triples)

    # -- Query helpers -----------------------------------------------------

    def query_run(self, run_id: str) -> list:
        """Get all triples for a benchmark run."""
        if self.store.doc_graph is None:
            return []
        head_triples = self.store.query_triples(head=run_id)
        tail_triples = self.store.query_triples(tail=run_id)
        return list(head_triples) + list(tail_triples)

    def query_by_factor(self, factor_name: str, level: str) -> list:
        """Retrieve all question triples with a specific factor level."""
        relation = f"factor_{factor_name}"
        return list(self.store.query_triples(relation=relation, tail=level))

    def query_question(self, doe_qid: str) -> list:
        """Get all triples for a specific DOE question."""
        if self.store.doc_graph is None:
            return []
        return list(self.store.query_triples(head=doe_qid))

    # -- Agentic harness tracing (WS1-WS7) -----------------------------------

    def trace_gateway_decision(
        self,
        run_id: str,
        tool_name: str,
        action: str,
        reason: str,
        gms_score: float | None = None,
        mock_plausibility_geodesic: float | None = None,
    ):
        """Record a tool gateway interception decision."""
        import uuid
        gw_id = f"gw_{uuid.uuid4().hex[:8]}"
        triples = [
            ExtractedTriple(head=gw_id, relation="is_type",
                            tail="gateway_decision"),
            ExtractedTriple(head=gw_id, relation="run", tail=run_id),
            ExtractedTriple(head=gw_id, relation="tool", tail=tool_name),
            ExtractedTriple(head=gw_id, relation="action", tail=action),
            ExtractedTriple(head=gw_id, relation="reason",
                            tail=reason[:200]),
            ExtractedTriple(head=gw_id, relation="timestamp",
                            tail=time.strftime("%Y-%m-%d %H:%M:%S")),
        ]
        if gms_score is not None:
            triples.append(ExtractedTriple(
                head=gw_id, relation="gms_score",
                tail=f"{gms_score:.4f}",
                numeric_value=gms_score))
        if mock_plausibility_geodesic is not None:
            triples.append(ExtractedTriple(
                head=gw_id, relation="mock_plausibility_geodesic",
                tail=f"{mock_plausibility_geodesic:.4f}",
                numeric_value=mock_plausibility_geodesic))
        self._learn(triples)

    def trace_policy_decision(
        self,
        run_id: str,
        action: str,
        allowed: bool,
        blocking_rules: list[str],
        gms_scores: dict[str, float],
    ):
        """Record a policy engine pre-execution decision."""
        import uuid
        pd_id = f"pd_{uuid.uuid4().hex[:8]}"
        triples = [
            ExtractedTriple(head=pd_id, relation="is_type",
                            tail="policy_decision"),
            ExtractedTriple(head=pd_id, relation="run", tail=run_id),
            ExtractedTriple(head=pd_id, relation="action", tail=action),
            ExtractedTriple(head=pd_id, relation="allowed",
                            tail=str(allowed).lower()),
            ExtractedTriple(head=pd_id, relation="timestamp",
                            tail=time.strftime("%Y-%m-%d %H:%M:%S")),
        ]
        for rule in blocking_rules:
            triples.append(ExtractedTriple(
                head=pd_id, relation="blocked_by_rule", tail=rule))
        for name, score in gms_scores.items():
            triples.append(ExtractedTriple(
                head=pd_id, relation=f"gms_{name}",
                tail=f"{score:.4f}", numeric_value=score))
        self._learn(triples)

    def trace_state_transition(
        self,
        run_id: str,
        from_state: str,
        to_state: str,
    ):
        """Record a run state machine transition."""
        triples = [
            ExtractedTriple(head=run_id, relation="transitioned_from",
                            tail=from_state),
            ExtractedTriple(head=run_id, relation="transitioned_to",
                            tail=to_state),
            ExtractedTriple(head=run_id, relation="transition_timestamp",
                            tail=time.strftime("%Y-%m-%d %H:%M:%S")),
        ]
        self._learn(triples)

    def trace_regression_promotion(
        self,
        run_id: str,
        scenario_id: str,
        source_question_id: str,
        failure_code: str,
    ):
        """Record promotion of a failed question to a regression scenario."""
        triples = [
            ExtractedTriple(head=scenario_id, relation="is_type",
                            tail="test_scenario"),
            ExtractedTriple(head=scenario_id, relation="promoted_from_run",
                            tail=run_id),
            ExtractedTriple(head=scenario_id, relation="source_question",
                            tail=source_question_id),
            ExtractedTriple(head=scenario_id, relation="failure_code",
                            tail=failure_code),
            ExtractedTriple(head=scenario_id, relation="promoted_at",
                            tail=time.strftime("%Y-%m-%d %H:%M:%S")),
        ]
        self._learn(triples)

    def trace_run_comparison(
        self,
        run_a_id: str,
        run_b_id: str,
        accuracy_delta: float,
        n_regressions: int,
        n_improvements: int,
        semantic_drift: float,
    ):
        """Record a run-to-run comparison result."""
        import uuid
        cmp_id = f"cmp_{uuid.uuid4().hex[:8]}"
        triples = [
            ExtractedTriple(head=cmp_id, relation="is_type",
                            tail="run_comparison"),
            ExtractedTriple(head=cmp_id, relation="run_a", tail=run_a_id),
            ExtractedTriple(head=cmp_id, relation="run_b", tail=run_b_id),
            ExtractedTriple(head=cmp_id, relation="accuracy_delta",
                            tail=f"{accuracy_delta:.4f}",
                            numeric_value=accuracy_delta),
            ExtractedTriple(head=cmp_id, relation="n_regressions",
                            tail=str(n_regressions),
                            numeric_value=float(n_regressions)),
            ExtractedTriple(head=cmp_id, relation="n_improvements",
                            tail=str(n_improvements),
                            numeric_value=float(n_improvements)),
            ExtractedTriple(head=cmp_id, relation="semantic_drift",
                            tail=f"{semantic_drift:.4f}",
                            numeric_value=semantic_drift),
            ExtractedTriple(head=cmp_id, relation="timestamp",
                            tail=time.strftime("%Y-%m-%d %H:%M:%S")),
        ]
        self._learn(triples)

    def trace_tiered_release_gate(
        self,
        run_id: str,
        tier_name: str,
        passed: bool,
        checks: dict[str, dict],
        blocking_reasons: list[str],
    ):
        """Record a risk-tiered release gate evaluation."""
        gate_id = f"tier_gate_{run_id}_{tier_name}"
        triples = [
            ExtractedTriple(head=gate_id, relation="is_type",
                            tail="risk_tier_gate"),
            ExtractedTriple(head=gate_id, relation="run", tail=run_id),
            ExtractedTriple(head=gate_id, relation="tier", tail=tier_name),
            ExtractedTriple(head=gate_id, relation="passed",
                            tail=str(passed).lower()),
            ExtractedTriple(head=gate_id, relation="timestamp",
                            tail=time.strftime("%Y-%m-%d %H:%M:%S")),
        ]
        for check_name, detail in checks.items():
            actual = detail.get("actual", 0.0)
            triples.append(ExtractedTriple(
                head=gate_id,
                relation=f"check_{check_name}",
                tail=f"{actual:.4f}",
                numeric_value=float(actual),
            ))
        for reason in blocking_reasons:
            triples.append(ExtractedTriple(
                head=gate_id, relation="blocked_by", tail=reason[:100]))
        self._learn(triples)

    # -- Performance profiling tracing -----------------------------------------

    def trace_performance_record(
        self,
        run_id: str,
        tool_name: str,
        latency_ms: float,
        sla_tension: float,
        within_sla: bool,
    ):
        """Record a performance measurement as GMS triples."""
        import uuid
        perf_id = f"perf_{uuid.uuid4().hex[:8]}"
        triples = [
            ExtractedTriple(head=perf_id, relation="is_type",
                            tail="performance_profile"),
            ExtractedTriple(head=perf_id, relation="run", tail=run_id),
            ExtractedTriple(head=perf_id, relation="tool", tail=tool_name),
            ExtractedTriple(head=perf_id, relation="latency_ms",
                            tail=f"{latency_ms:.2f}",
                            numeric_value=latency_ms),
            ExtractedTriple(head=perf_id, relation="sla_tension",
                            tail=f"{sla_tension:.4f}",
                            numeric_value=sla_tension),
            ExtractedTriple(head=perf_id, relation="within_sla",
                            tail=str(within_sla).lower()),
            ExtractedTriple(head=perf_id, relation="timestamp",
                            tail=time.strftime("%Y-%m-%d %H:%M:%S")),
        ]
        self._learn(triples)

    # -- Cost tracking tracing -------------------------------------------------

    def trace_cost_record(
        self,
        run_id: str,
        tool_name: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        cumulative_cost_usd: float,
        budget_tension: float,
        within_budget: bool,
    ):
        """Record a cost/token event as GMS triples."""
        import uuid
        cost_id = f"cost_{uuid.uuid4().hex[:8]}"
        triples = [
            ExtractedTriple(head=cost_id, relation="is_type",
                            tail="cost_record"),
            ExtractedTriple(head=cost_id, relation="run", tail=run_id),
            ExtractedTriple(head=cost_id, relation="tool", tail=tool_name),
            ExtractedTriple(head=cost_id, relation="input_tokens",
                            tail=str(input_tokens),
                            numeric_value=float(input_tokens)),
            ExtractedTriple(head=cost_id, relation="output_tokens",
                            tail=str(output_tokens),
                            numeric_value=float(output_tokens)),
            ExtractedTriple(head=cost_id, relation="cost_usd",
                            tail=f"{cost_usd:.6f}",
                            numeric_value=cost_usd),
            ExtractedTriple(head=cost_id, relation="cumulative_cost_usd",
                            tail=f"{cumulative_cost_usd:.6f}",
                            numeric_value=cumulative_cost_usd),
            ExtractedTriple(head=cost_id, relation="budget_tension",
                            tail=f"{budget_tension:.4f}",
                            numeric_value=budget_tension),
            ExtractedTriple(head=cost_id, relation="within_budget",
                            tail=str(within_budget).lower()),
            ExtractedTriple(head=cost_id, relation="timestamp",
                            tail=time.strftime("%Y-%m-%d %H:%M:%S")),
        ]
        self._learn(triples)

    # -- Persistence & query ---------------------------------------------------

    def save(self):
        if self.learner:
            self.learner.wait_for_training(timeout=120.0)
        self.store.save()

    def stats(self) -> dict:
        return self.store.stats()
