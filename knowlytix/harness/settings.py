# SPDX-License-Identifier: Apache-2.0
"""Flat env-var-overrideable settings for the GMSH harness.

Mirrors the user-facing subset of the three legacy dataclasses that own
harness configuration today:

- :class:`harness.core.HarnessConfig` (turn limits, context, memory, paths)
- :class:`knowlytix.harness.testing.config.DOEHarnessConfig` (DOE knobs, SLAs, feature flags)
- :class:`knowlytix.harness.testing.parallel.ParallelConfig` (concurrency and retries)

The legacy dataclasses remain in place for YAML/CLI workflows; wheel
users read :class:`GMSHSettings` at runtime to override the tunables
through environment variables with the ``GMSH_`` prefix.

See ``distribution/config_prereq_plan.md`` §Domain 5.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from pydantic import Field
from pydantic_settings import BaseSettings

from knowlytix.core._caps import Unlimited, apply_ceiling, get_numeric_cap
from knowlytix.core._settings_base import settings_config

if TYPE_CHECKING:
    from knowlytix.harness.core import HarnessConfig
    from knowlytix.harness.testing.config import DOEHarnessConfig
    from knowlytix.harness.testing.parallel import ParallelConfig


class GMSHSettings(BaseSettings):
    """Flat env-var-overrideable settings for the ``gmsh`` harness.

    Env prefix: ``GMSH_``. Covers DOE benchmark tunables, parallelism,
    harness resource limits, storage paths, and feature flags. The
    legacy dataclasses (``HarnessConfig``, ``DOEHarnessConfig``,
    ``ParallelConfig``) stay as the authoritative source for YAML/CLI
    flows; this class is the flat view wheel users talk to.

    Example::

        export GMSH_MAX_WORKERS=8
        export GMSH_DOE_N_RUNS=64
        export GMSH_ENABLE_TRACING=false

        from knowlytix.harness.settings import GMSHSettings
        s = GMSHSettings()
    """

    model_config = settings_config(env_prefix="GMSH_")

    # ---------------------------------------------------------------
    # DOE benchmark knobs (mirrors DOEHarnessConfig user-facing subset)
    # ---------------------------------------------------------------
    doe_n_runs: int = Field(default=32, gt=0)
    doe_seed: int = 42
    doe_max_per_category: int = Field(default=10, gt=0)
    doe_ingest_epochs: int = Field(default=200, gt=0)

    # SLAs & budgets (must be positive, else downstream logic misbehaves).
    doe_sla_latency_ms: float = Field(default=5000.0, gt=0)
    doe_cost_budget_usd: float = Field(default=10.0, ge=0)

    # Specialized testing sizes
    doe_stateful_session_count: int = Field(default=5, gt=0)
    # None = use the calibrated default from store calibration.json. Float
    # = explicit override (the underlying field was named
    # hallucination_geodesic_threshold; renamed to hallucination_tau_override
    # when the calibration surface was generalized).
    doe_hallucination_threshold: float | None = Field(default=None, ge=0)

    # Dashboard port for live tracing UI. 1–65535 per RFC; reject 0
    # (OS-assigned) and out-of-range values up front.
    live_dashboard_port: int = Field(default=8765, ge=1, le=65535)

    # ---------------------------------------------------------------
    # Parallelism (mirrors ParallelConfig)
    # ---------------------------------------------------------------
    # ThreadPoolExecutor crashes on max_workers <= 0; enforce ≥1 here
    # so the misconfiguration fails at Settings construction with a
    # clear Pydantic error, not inside a worker thread.
    max_workers: int = Field(default=4, ge=1)
    question_timeout_s: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    retry_delay_s: float = Field(default=1.0, ge=0)

    # ---------------------------------------------------------------
    # Harness resource limits (mirrors HarnessConfig)
    # ---------------------------------------------------------------
    max_turns: int = Field(default=30, gt=0)
    truncate_result_at: int = Field(default=8000, gt=0)
    max_message_tokens: int = Field(default=180_000, gt=0)
    memory_context_lines: int = Field(default=30, ge=0)

    # ---------------------------------------------------------------
    # Storage paths — defaults relative to CWD so wheel users running
    # outside the repo get a sensible location.
    # ---------------------------------------------------------------
    stores_dir: Path = Path("./gms_stores")
    tracing_dir: Path = Path("./doe_tracing_store")
    runs_dir: Path = Path("./runs")
    campaigns_dir: Path = Path("./campaigns")
    session_store_path: Path = Path("./harness_session_store")

    # ---------------------------------------------------------------
    # Feature flags (subset of DOEHarnessConfig.enable_* fields)
    # ---------------------------------------------------------------
    enable_tracing: bool = True
    enable_typed_verdicts: bool = False
    enable_provenance: bool = False
    enable_gateway: bool = False
    enable_policy_engine: bool = False
    enable_scenario_registry: bool = False
    enable_process_evaluation: bool = False
    enable_recovery_evaluation: bool = False
    enable_interaction_evaluation: bool = False
    enable_severity_classification: bool = False
    enable_performance_profiling: bool = False
    enable_cost_tracking: bool = False
    enable_stateful_testing: bool = False
    enable_hallucination_testing: bool = False
    enable_calibration_testing: bool = False
    enable_multi_agent: bool = False
    enable_cross_document: bool = False
    enable_disambiguation_testing: bool = False
    enable_invariance_testing: bool = False
    enable_streaming_testing: bool = False
    enable_live_dashboard: bool = False

    @classmethod
    @functools.cache
    def _enable_flag_fields(cls) -> tuple[str, ...]:
        """Names of the `enable_*` boolean fields on this class.

        Introspected via :attr:`model_fields` so a future ``enable_*``
        field added to the class is automatically picked up by
        :meth:`from_license` without touching this method.  Result is
        cached — ``model_fields`` is frozen after class definition, so
        the tuple is stable for the whole process lifetime.
        """
        return tuple(
            name for name in cls.model_fields if name.startswith("enable_")
        )

    @classmethod
    def from_license(
        cls,
        caps: "Mapping[str, int | bool | Unlimited] | None" = None,
    ) -> "GMSHSettings":
        """Return a :class:`GMSHSettings` with license ceilings applied.

        **Precondition**: ``caps`` must be a validated caps dict (either
        passed explicitly or fetched from compiled ``knowlytix-core`` via
        :func:`gms._caps.get_caps` when ``caps`` is ``None``).  A
        missing required claim raises ``KeyError`` (fail-closed,
        Invariant 7).

        The ceiling rule itself lives in compiled ``knowlytix-core``: this
        classmethod is the open-source hook that pulls caps from there
        and applies them.  Clamps:
        - ``doe_n_runs`` against ``caps["max_benchmark_runs"]``
        - ``max_workers`` against ``caps["max_workers"]``

        Boolean setter-shaped claim (Invariant 6):
        ``enable_advanced_features=False`` forces every ``enable_*``
        feature flag on this class to ``False`` regardless of env-var
        setting.  License ``False`` cannot be flipped to runtime ``True``
        by any customer env var.  ``True`` defers to user config.
        """
        if caps is None:
            # `get_caps()` returns an immutable MappingProxyType.  We never
            # mutate `caps` below (all reads are subscripts into a local
            # `overrides` dict), so skipping the dict copy saves an 8-key
            # allocation on every call.
            from knowlytix.core._caps import get_caps

            caps = get_caps()

        inst = cls()
        overrides: dict[str, object] = {}

        # --- Numeric ceilings --------------------------------------------
        clamped_runs = apply_ceiling(
            "doe_n_runs",
            inst.doe_n_runs,
            get_numeric_cap(caps, "max_benchmark_runs"),
        )
        if clamped_runs != inst.doe_n_runs:
            overrides["doe_n_runs"] = clamped_runs

        clamped_workers = apply_ceiling(
            "max_workers",
            inst.max_workers,
            get_numeric_cap(caps, "max_workers"),
        )
        if clamped_workers != inst.max_workers:
            overrides["max_workers"] = clamped_workers

        # --- Setter-shaped boolean: enable_advanced_features -------------
        # License `false` forces every `enable_*` flag to False regardless
        # of env var (Invariant 6 of license_capability_plan.md).
        # Enumerated dynamically via introspection so a new `enable_*`
        # field added to this class is automatically gated without
        # touching this code.  Introspection happens at call time
        # (not at module import) so there's no "empty tuple" footgun
        # if `from_license` is somehow invoked mid-module-definition.
        # Uses `is not True` (not `is False`) so a tampered caps dict that
        # smuggles in a non-bool truthy value (e.g. 1, "yes") still gates
        # features off — strict fail-closed on non-validated input.
        if caps["enable_advanced_features"] is not True:
            for field_name in cls._enable_flag_fields():
                if getattr(inst, field_name) is not False:
                    overrides[field_name] = False

        if not overrides:
            return inst
        return inst.model_copy(update=overrides)

    @classmethod
    def from_legacy(
        cls,
        doe: "DOEHarnessConfig | None" = None,
        parallel: "ParallelConfig | None" = None,
        harness: "HarnessConfig | None" = None,
    ) -> "GMSHSettings":
        """Project any combination of legacy configs into flat settings.

        Any argument omitted uses the corresponding class defaults.
        """

        kwargs: dict[str, object] = {}

        if doe is not None:
            kwargs.update(
                {
                    "doe_n_runs": doe.n_runs,
                    "doe_seed": doe.seed,
                    "doe_max_per_category": doe.max_per_category,
                    "doe_ingest_epochs": doe.ingest_epochs,
                    "doe_sla_latency_ms": doe.sla_latency_ms,
                    "doe_cost_budget_usd": doe.cost_budget_usd,
                    "doe_stateful_session_count": doe.stateful_session_count,
                    "doe_hallucination_threshold": doe.hallucination_tau_override,
                    "live_dashboard_port": doe.live_dashboard_port,
                    "tracing_dir": Path(doe.tracing_dir),
                    "stores_dir": Path(doe.stores_dir),
                    "enable_tracing": doe.enable_tracing,
                    "enable_typed_verdicts": doe.enable_typed_verdicts,
                    "enable_provenance": doe.enable_provenance,
                    "enable_gateway": doe.enable_gateway,
                    "enable_policy_engine": doe.enable_policy_engine,
                    "enable_scenario_registry": doe.enable_scenario_registry,
                    "enable_process_evaluation": doe.enable_process_evaluation,
                    "enable_recovery_evaluation": doe.enable_recovery_evaluation,
                    "enable_interaction_evaluation": doe.enable_interaction_evaluation,
                    "enable_severity_classification": doe.enable_severity_classification,
                    "enable_performance_profiling": doe.enable_performance_profiling,
                    "enable_cost_tracking": doe.enable_cost_tracking,
                    "enable_stateful_testing": doe.enable_stateful_testing,
                    "enable_hallucination_testing": doe.enable_hallucination_testing,
                    "enable_calibration_testing": doe.enable_calibration_testing,
                    "enable_multi_agent": doe.enable_multi_agent,
                    "enable_cross_document": doe.enable_cross_document,
                    "enable_disambiguation_testing": doe.enable_disambiguation_testing,
                    "enable_invariance_testing": doe.enable_invariance_testing,
                    "enable_streaming_testing": doe.enable_streaming_testing,
                    "enable_live_dashboard": doe.enable_live_dashboard,
                }
            )
            if doe.runs_dir:
                kwargs["runs_dir"] = Path(doe.runs_dir)
            if doe.campaigns_dir:
                kwargs["campaigns_dir"] = Path(doe.campaigns_dir)

        if parallel is not None:
            kwargs.update(
                {
                    "max_workers": parallel.max_workers,
                    "question_timeout_s": parallel.timeout_per_question_s,
                    "max_retries": parallel.max_retries,
                    "retry_delay_s": parallel.retry_delay_s,
                }
            )

        if harness is not None:
            kwargs.update(
                {
                    "max_turns": harness.max_turns,
                    "truncate_result_at": harness.truncate_result_at,
                    "max_message_tokens": harness.max_message_tokens,
                    "memory_context_lines": harness.memory_context_lines,
                    "stores_dir": Path(harness.stores_dir),
                    "session_store_path": Path(harness.session_store_path),
                }
            )

        return cls(**kwargs)
