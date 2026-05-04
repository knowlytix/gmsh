# SPDX-License-Identifier: Apache-2.0
"""
GraphDOE: Graph-Verified DOE Testing for LLMs and Agentic Systems.

Combines graph-constructed ground truth (provably correct answers) with
Design of Experiments methodology (systematic factor variation) to test
any LLM or agentic system and statistically attribute failure causes.

Quick start::

    from knowlytix.core.llm import get_llm
    from knowlytix.harness.graphdoe import DOEBenchmark

    bench = DOEBenchmark("report.md", factor_group="query_core", n_runs=32)
    result = bench.run(llm=get_llm())
    result.analyze()

With custom evaluator::

    def my_agent(question, context):
        return answer

    result = bench.run(evaluator=my_agent)
"""

__version__ = "0.2.0"

from .doe_benchmark import DOEBenchmark
from .design_matrix import DesignMatrix
from .question_rephraser import QuestionRephraser, DOEQuestion
from .analysis import DOEAnalyzer
from .results import DOEBenchmarkResult
from .factor_loader import FactorLoader
from .paper_api import design
