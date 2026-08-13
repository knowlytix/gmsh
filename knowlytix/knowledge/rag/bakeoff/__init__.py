# SPDX-License-Identifier: Apache-2.0
"""Binder Bake-Off — a controlled comparison of NL->query-triple *binder*
configurations over one GMS substrate.

The surrounding RAG pipeline (retrieval, relevance, verify, assemble, synthesis)
is held fixed; only the mechanism that turns a natural-language question into a
bound query triple varies across arms. This package adds the capabilities the
comparison needs that the base :mod:`knowlytix.knowledge.rag` does not already
provide, kept in a separate module so the shared ``eval.py``/``pipeline.py``
signatures other applications depend on are not disturbed:

* :mod:`realized_factors` — measure the *realized* level of each DOE presentation
  factor in a rephrased question, so analysis keys on the coordinate the text
  actually occupies rather than the one the design requested.
* :mod:`roundtrip` — filter generated ``(question, hops)`` pairs to those that
  execute against the graph oracle, with a vanished-cell ledger flagging base
  facts/probes the graph cannot support under any phrasing.

Everything else in the spec (arms A/B, DOE generation, accept-cohort calibration,
factor attribution, the GMS primitives) is reused from the existing library, not
reimplemented here.
"""

from __future__ import annotations

from .arms import (
    AliasHybridParser,
    AliasTableResolver,
    ExemplarFewShotParser,
    ExemplarIndex,
    calibrate_alias_resolver,
)
from .conformal import ABSTAIN_LABEL, ConformalGate
from .decision import DEFAULT_CEILINGS, decide
from .harness import Arm, BinderPrediction, bakeoff, run_arm
from .holdout import (
    DISJOINT_HOLDOUT_FACTOR_NAMES,
    build_holdout_manifest,
    load_materialized_holdout,
    write_manifest,
)
from .metrics import (
    PATCH_COST_STEPS,
    BinderResult,
    metric_report,
    novel_entity_report,
    patch_cost,
    slice_by_factor,
    synthetic_vs_real_gap,
)
from .realized_factors import (
    RealizedFactorScorer,
    geometric_level_measurer,
    length_measurer,
    paraphrase_depth_measurer,
)
from .roundtrip import RoundTripReport, graph_execute, round_trip_filter
from .schema_guard import SchemaGuard

__all__ = [
    "RealizedFactorScorer",
    "length_measurer",
    "paraphrase_depth_measurer",
    "geometric_level_measurer",
    "round_trip_filter",
    "RoundTripReport",
    "graph_execute",
    "build_holdout_manifest",
    "write_manifest",
    "load_materialized_holdout",
    "DISJOINT_HOLDOUT_FACTOR_NAMES",
    "ConformalGate",
    "ABSTAIN_LABEL",
    "BinderResult",
    "metric_report",
    "novel_entity_report",
    "slice_by_factor",
    "synthetic_vs_real_gap",
    "patch_cost",
    "PATCH_COST_STEPS",
    "Arm",
    "BinderPrediction",
    "run_arm",
    "bakeoff",
    "SchemaGuard",
    "ExemplarIndex",
    "ExemplarFewShotParser",
    "AliasTableResolver",
    "AliasHybridParser",
    "calibrate_alias_resolver",
    "DEFAULT_CEILINGS",
    "decide",
]
