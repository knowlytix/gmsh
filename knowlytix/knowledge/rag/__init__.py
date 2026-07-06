# SPDX-License-Identifier: Apache-2.0
"""GEODE graph-RAG — triple-mediated retrieval over a GMS expert store.

Prose and structured questions alike are translated into query triples, answered
through the GMS (with provenance-linked source spans), and synthesized by a
user-chosen LLM. Bank-grade by default: dense retrieval is off, answers are
GMS-grounded, and the pipeline abstains rather than guess. See
``GEODE_RAG_DESIGN.md``.
"""

from knowlytix.knowledge.rag.assemble import Assembler
from knowlytix.knowledge.rag.binding import BoundTriple, TripleBinder
from knowlytix.knowledge.rag.config import RagConfig
from knowlytix.knowledge.rag.coverage import (
    CoverageReport,
    GraphCoverage,
    RegionCoverage,
    coverage_report,
    graph_coverage,
)
from knowlytix.knowledge.rag.dense import (
    DenseSpan,
    InMemoryVectorBackend,
    VectorBackend,
    build_spans,
)
from knowlytix.knowledge.rag.eval import (
    EvalCase,
    EvalReport,
    GroundingCheck,
    GroundingFilter,
    RetrievalCaseResult,
    benchmark_query_parse,
    benchmark_retrieval,
    calibrate_accept_threshold,
    calibrate_bind_threshold,
    evaluate,
    filter_grounding_preserving,
    grounding_preserved,
)
from knowlytix.knowledge.rag.admissibility_filter import filter_admissible_facts
from knowlytix.knowledge.rag.extraction import Extraction
from knowlytix.knowledge.rag.pipeline import RagAnswer, RagPipeline
from knowlytix.knowledge.rag.relevance import RelevanceGate
from knowlytix.knowledge.rag.label_classifier import GeometricLabelClassifier
from knowlytix.knowledge.rag.query_triples import (
    QueryTriple,
    QueryTripleExtractor,
    is_var,
    schema_from_store,
)
from knowlytix.knowledge.rag.retrieve import (
    RetrievedFact,
    RetrieveResult,
    Retriever,
)
from knowlytix.knowledge.rag.value_polarity import (
    PolarityCuts, ValuePolarityChecker, polarity_tension_cohort)
from knowlytix.knowledge.rag.governed import (
    GovernedRetriever,
    RetrievalContract,
    SensitivityMap,
    ProtectedProbe,
    PolarityDisclosureGuard,
    ClassifierDisclosureGuard,
    DEFAULT_STANCE_VALUES,
    GovernedFact,
    DeniedFact,
    DisclosureFinding,
    RetrievalAuditRecord,
    GovernedResult,
    SENSITIVITY_ORDER,
    REDACTORS,
)
from knowlytix.knowledge.rag.verify import (
    AnswerVerifier,
    ClaimVerdict,
    VerifyReport,
)

__all__ = [
    "RagConfig",
    "RagPipeline",
    "RagAnswer",
    "Extraction",
    "RelevanceGate",
    "filter_admissible_facts",
    "GeometricLabelClassifier",
    "QueryTriple",
    "QueryTripleExtractor",
    "is_var",
    "TripleBinder",
    "BoundTriple",
    "Retriever",
    "RetrieveResult",
    "RetrievedFact",
    "Assembler",
    "AnswerVerifier",
    "ClaimVerdict",
    "VerifyReport",
    "ValuePolarityChecker",
    "PolarityCuts",
    "polarity_tension_cohort",
    "GovernedRetriever",
    "RetrievalContract",
    "SensitivityMap",
    "ProtectedProbe",
    "PolarityDisclosureGuard",
    "ClassifierDisclosureGuard",
    "DEFAULT_STANCE_VALUES",
    "GovernedFact",
    "DeniedFact",
    "DisclosureFinding",
    "RetrievalAuditRecord",
    "GovernedResult",
    "SENSITIVITY_ORDER",
    "REDACTORS",
    "coverage_report",
    "CoverageReport",
    "GraphCoverage",
    "graph_coverage",
    "RegionCoverage",
    "schema_from_store",
    "EvalCase",
    "EvalReport",
    "evaluate",
    "calibrate_bind_threshold",
    "calibrate_accept_threshold",
    "benchmark_query_parse",
    "RetrievalCaseResult",
    "benchmark_retrieval",
    "GroundingCheck",
    "grounding_preserved",
    "GroundingFilter",
    "filter_grounding_preserving",
    "VectorBackend",
    "InMemoryVectorBackend",
    "DenseSpan",
    "build_spans",
]
