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

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _dist_version

try:
    __version__ = _dist_version("knowlytix-harness")
except _PkgNotFound:  # source tree (dev install is the `gms` distribution)
    __version__ = "0.0.0.dev0"

from .analysis import DOEAnalyzer
from .design_matrix import DesignMatrix
from .doe_benchmark import DOEBenchmark
from .factor_loader import FactorLoader
from .paper_api import design
from .question_rephraser import DOEQuestion, QuestionRephraser
from .results import DOEBenchmarkResult

# Re-export canonical scorers when knowlytix.benchmark.eval.* is reachable.
# This is the normal case inside the gms source repository. A vendored / standalone
# graphdoe without it on sys.path is still fully usable — callers can import
# the scorers from knowlytix.benchmark.eval.scorers directly when needed.
try:
    from knowlytix.benchmark.eval.scorers import (  # noqa: E402,F401
        ScoreResult,
        score_answer,
        score_path_validity,
    )
except ImportError:
    pass
