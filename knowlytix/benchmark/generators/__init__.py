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
from knowlytix.benchmark.generators.multi_hop import MultiHopGenerator
from knowlytix.benchmark.generators.relation_recall import (
    RelationAbsentProbeGenerator,
    RelationChainGenerator,
)
from knowlytix.benchmark.generators.threshold import ThresholdGenerator


def default_generators():
    return [
        ExactRecallGenerator(),
        ThresholdGenerator(),
        CrossReferenceGenerator(),
        ContradictionGenerator(),
        MultiHopGenerator(),
        CountingGenerator(),
    ]
