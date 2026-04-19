# SPDX-License-Identifier: Apache-2.0
from knowlytix.bench.generators.base import QuestionGenerator, GeneratedQuestion, ANSWER_FORMATS
from knowlytix.bench.generators.exact_recall import ExactRecallGenerator
from knowlytix.bench.generators.threshold import ThresholdGenerator
from knowlytix.bench.generators.cross_reference import CrossReferenceGenerator
from knowlytix.bench.generators.contradiction import ContradictionGenerator
from knowlytix.bench.generators.multi_hop import MultiHopGenerator
from knowlytix.bench.generators.counting import CountingGenerator


def default_generators():
    return [
        ExactRecallGenerator(),
        ThresholdGenerator(),
        CrossReferenceGenerator(),
        ContradictionGenerator(),
        MultiHopGenerator(),
        CountingGenerator(),
    ]
