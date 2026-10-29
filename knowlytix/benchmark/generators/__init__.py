# SPDX-License-Identifier: Apache-2.0
from knowlytix.benchmark.generators.base import (
    ANSWER_FORMATS,
    GeneratedQuestion,
    QuestionGenerator,
)
from knowlytix.benchmark.generators.contradiction import ContradictionGenerator
from knowlytix.benchmark.generators.counting import CountingGenerator
from knowlytix.benchmark.generators.cross_reference import CrossReferenceGenerator
from knowlytix.benchmark.generators.exact_recall import ExactRecallGenerator
from knowlytix.benchmark.generators.fact_recall import FactRecallGenerator
from knowlytix.benchmark.generators.generic import (
    GENERIC_GENERATORS,
    AbsenceDetectionGenerator,
    BoundaryGenerator,
    ComparativeDiscriminationGenerator,
    CrossTableAggregationGenerator,
    FactCompletionGenerator,
    GovernancePolicyGenerator,
    NumericComputationGenerator,
    PathConsistencyGenerator,
    PlausibilityGenerator,
    ProceduralOrderingGenerator,
    RankingGenerator,
    RecoveryGenerator,
)
from knowlytix.benchmark.generators.multi_hop import MultiHopGenerator
from knowlytix.benchmark.generators.relation_recall import (
    RelationAbsentProbeGenerator,
    RelationChainGenerator,
)
from knowlytix.benchmark.generators.threshold import ThresholdGenerator


def default_generators():
    # Full generator set. Previously only the first six were wired, so the
    # benchmark (and graphDOE base-query generation built on it) covered a small
    # slice of the query space and never emitted the ranking / numeric /
    # aggregation / absence / relation categories. Include every imported
    # generator so the base coverage is complete.
    return [
        ExactRecallGenerator(),
        ThresholdGenerator(),
        CrossReferenceGenerator(),
        ContradictionGenerator(),
        MultiHopGenerator(),
        CountingGenerator(),
        FactRecallGenerator(),
        RelationAbsentProbeGenerator(),
        RelationChainGenerator(),
        *[g() for g in GENERIC_GENERATORS],
    ]
