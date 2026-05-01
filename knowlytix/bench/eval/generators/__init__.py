# SPDX-License-Identifier: Apache-2.0
from knowlytix.bench.eval.generators.base import QuestionGenerator, GeneratedQuestion
from knowlytix.bench.eval.generators.exact_recall import ExactRecallGenerator
from knowlytix.bench.eval.generators.threshold import ThresholdGenerator
from knowlytix.bench.eval.generators.cross_reference import CrossReferenceGenerator
from knowlytix.bench.eval.generators.contradiction import ContradictionGenerator
from knowlytix.bench.eval.generators.multi_hop import MultiHopGenerator
from knowlytix.bench.eval.generators.counting import CountingGenerator


def default_generators():
    """Return one instance of each generator."""
    return [
        ExactRecallGenerator(),
        ThresholdGenerator(),
        CrossReferenceGenerator(),
        ContradictionGenerator(),
        MultiHopGenerator(),
        CountingGenerator(),
    ]
