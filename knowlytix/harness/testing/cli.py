# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for DOE-GMS benchmarks.

Usage::

    # Quick run with defaults
    python -m knowlytix.harness.testing report.md

    # Custom factors and target
    python -m knowlytix.harness.testing report.md \\
        --factor-group query_core \\
        --n-runs 64 \\
        --target-model claude-sonnet-4-20250514 \\
        --context-mode oracle_section \\
        --output results.json

    # From YAML config
    python -m knowlytix.harness.testing --config doe_config.yaml

    # Agent mode (uses Harness with tools)
    python -m knowlytix.harness.testing report.md --target-type agent

    # Agentic harness subcommands
    python -m knowlytix.harness.testing runs list
    python -m knowlytix.harness.testing runs show <run_id>
    python -m knowlytix.harness.testing replay <manifest.yaml>
    python -m knowlytix.harness.testing regression promote <run_id>
    python -m knowlytix.harness.testing regression list
    python -m knowlytix.harness.testing compare <run_a_result.json> <run_b_result.json>
"""

from __future__ import annotations

import argparse
import json


# ---------------------------------------------------------------------------
# Shared argument builders
# ---------------------------------------------------------------------------

def _add_run_arguments(parser: argparse.ArgumentParser):
    """Add the standard DOE benchmark arguments to a parser."""
    parser.add_argument(
        "markdown", nargs="?", default="",
        help="Path to markdown document to test against",
    )
    parser.add_argument(
        "--config", default="",
        help="YAML config file (overrides other flags)",
    )

    # DOE
    parser.add_argument("--factor-group", default="quick_screen")
    parser.add_argument("--factor-names", nargs="*", default=[])
    parser.add_argument("--n-runs", type=int, default=32)
    parser.add_argument("--method", default="sobol",
                        choices=["sobol", "lhs", "grid"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rephrase-method", default="template",
                        choices=["template", "llm"])

    # Questions
    parser.add_argument("--max-per-category", type=int, default=10)
    parser.add_argument("--generators", nargs="*", default=[])

    # Target
    parser.add_argument("--target-type", default="llm",
                        choices=["llm", "agent"])
    parser.add_argument("--target-model", default="claude-sonnet-4-20250514")
    parser.add_argument("--context-mode", default="full",
                        choices=["full", "oracle_section", "noisy_section"])

    # Tracing
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--tracing-dir", default="doe_tracing_store")

    # Output
    parser.add_argument("--output", default="", help="Save results to JSON")
    parser.add_argument("--report", default="",
                        help="Generate HTML dashboard report at this path")
    parser.add_argument("--stores-dir", default="gms_stores")
    parser.add_argument("--device", default="")

    # GMSH features
    parser.add_argument("--typed-verdicts", action="store_true",
                        help="Enable GMSH typed claim verification")
    parser.add_argument("--provenance", action="store_true",
                        help="Enable document provenance tracking")
    parser.add_argument("--snapshot-dir", default="",
                        help="Directory for GMS store snapshots")
    parser.add_argument("--audit-report", default="",
                        help="Generate GMSH audit report at this path")
    parser.add_argument("--release-gate", nargs="*", default=[],
                        help="Release gate thresholds: accuracy=0.8 geo=0.5")
    parser.add_argument("--semantic-mode", default="off",
                        choices=["off", "fallback", "complement"],
                        help="LLM-as-judge semantic verification mode")
    parser.add_argument("--semantic-model", default="",
                        help="Model for semantic judge (default: target model)")

    # Risk-tiered gates (WS7)
    parser.add_argument("--risk-tiers", default="",
                        help="YAML file with risk tier profiles")
    parser.add_argument("--baseline-run", default="",
                        help="Baseline run ID for regression comparison")

    # Modes
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only run GMS baseline (no LLM)")


# ---------------------------------------------------------------------------
# Build config from args
# ---------------------------------------------------------------------------

def _build_config(args):
    """Build DOEHarnessConfig from parsed arguments."""
    from .config import DOEHarnessConfig

    if args.config:
        config = DOEHarnessConfig.from_yaml(args.config)
        if args.markdown:
            config.markdown_path = args.markdown
    else:
        if not args.markdown:
            raise ValueError("markdown path is required (or use --config)")
        config = DOEHarnessConfig(
            markdown_path=args.markdown,
            stores_dir=args.stores_dir,
            factor_group=args.factor_group,
            factor_names=args.factor_names,
            n_runs=args.n_runs,
            method=args.method,
            seed=args.seed,
            rephrase_method=args.rephrase_method,
            max_per_category=args.max_per_category,
            generators=args.generators,
            target_type=args.target_type,
            target_model=args.target_model,
            context_mode=args.context_mode,
            enable_tracing=not args.no_trace,
            tracing_dir=args.tracing_dir,
            device=args.device,
            enable_typed_verdicts=args.typed_verdicts,
            enable_provenance=args.provenance,
            snapshot_dir=args.snapshot_dir,
            semantic_mode=args.semantic_mode,
            semantic_model=args.semantic_model,
            risk_tier_profiles_path=getattr(args, "risk_tiers", ""),
            baseline_run_id=getattr(args, "baseline_run", ""),
        )
    return config


# ---------------------------------------------------------------------------
# Subcommand: run (default)
# ---------------------------------------------------------------------------

def _cmd_run(args):
    """Execute the standard DOE-GMS benchmark."""
    config = _build_config(args)

    from .benchmark import DOEGMSBenchmark

    bench = DOEGMSBenchmark(config)
    bench.ingest()

    if args.baseline_only:
        result = bench.run_gms_baseline()
    else:
        from .evaluators import make_evaluator
        evaluator = make_evaluator(
            target_type=config.target_type,
            target_model=config.target_model,
        )
        result = bench.run(evaluator=evaluator)

    # Print
    bench.print_results(result)

    # Analyze
    analyzer = None
    try:
        analyzer = bench.analyze(result)
        analyzer.print_report()
    except Exception as e:
        print(f"\n  Analysis skipped: {e}")

    # Save
    if args.output:
        result.save(args.output)
        print(f"\n  Results saved to {args.output}")

    bench.save_tracing()

    # Dashboard report
    if args.report:
        from .dashboard import generate_report
        generate_report(
            result,
            analyzer=analyzer,
            output_path=args.report,
            tracing=bench.tracing,
        )

    # GMSH audit report
    if args.audit_report:
        # Parse release gate thresholds from CLI
        gate_kwargs = {}
        for item in (args.release_gate or []):
            if "=" in item:
                k, v = item.split("=", 1)
                if k == "accuracy":
                    gate_kwargs["accuracy_threshold"] = float(v)
                elif k == "geo":
                    gate_kwargs["geo_confidence_threshold"] = float(v)
        bench.generate_audit_report(
            result,
            analyzer=analyzer,
            output_path=args.audit_report,
            **gate_kwargs,
        )

    return result


# ---------------------------------------------------------------------------
# Subcommand: runs list / runs show
# ---------------------------------------------------------------------------

def _cmd_runs(args):
    """Manage benchmark run records."""
    from .runs import RunManager, RunState

    runs_dir = getattr(args, "runs_dir", "") or "doe_runs"
    manager = RunManager(runs_dir)

    action = getattr(args, "runs_action", "list")

    if action == "list":
        state_filter = getattr(args, "state", None)
        state = RunState(state_filter) if state_filter else None
        records = manager.list_runs(state=state)

        if not records:
            print("  No runs found.")
            return

        print(f"\n  {'Run ID':<32} {'State':<12} {'Accuracy':>8} "
              f"{'Questions':>10} {'Created':<20}")
        print(f"  {'─' * 84}")
        for r in records:
            acc = f"{r.accuracy:.1%}" if r.accuracy > 0 else "—"
            qs = str(r.n_questions) if r.n_questions > 0 else "—"
            print(f"  {r.run_id:<32} {r.state.value:<12} {acc:>8} "
                  f"{qs:>10} {r.created_at:<20}")

    elif action == "show":
        run_id = getattr(args, "run_id", "")
        if not run_id:
            print("  ERROR: run_id required")
            return

        record = manager.get_run(run_id)
        if record is None:
            print(f"  Run not found: {run_id}")
            return

        print(f"\n  Run: {record.run_id}")
        print(f"  State: {record.state.value}")
        print(f"  Created: {record.created_at}")
        if record.started_at:
            print(f"  Started: {record.started_at}")
        if record.completed_at:
            print(f"  Completed: {record.completed_at}")
        if record.n_questions > 0:
            print(f"  Questions: {record.n_questions}")
            print(f"  Correct: {record.n_correct}")
            print(f"  Accuracy: {record.accuracy:.1%}")
        if record.result_path:
            print(f"  Result: {record.result_path}")
        if record.error:
            print(f"  Error: {record.error}")
        if record.scenario_ids:
            print(f"  Scenarios: {', '.join(record.scenario_ids)}")

        if record.state_history:
            print("\n  State History:")
            for ts, from_s, to_s in record.state_history:
                print(f"    {ts}: {from_s} -> {to_s}")


# ---------------------------------------------------------------------------
# Subcommand: replay
# ---------------------------------------------------------------------------

def _cmd_replay(args):
    """Replay a benchmark run from a manifest."""
    manifest_path = getattr(args, "manifest_path", "")
    if not manifest_path:
        print("  ERROR: manifest path required")
        return

    from .replay import ReplayManifest, replay_run

    manifest = ReplayManifest.from_yaml(manifest_path)
    print(f"  Replaying from manifest: {manifest.manifest_id}")
    print(f"  Source run: {manifest.source_run_id}")
    print(f"  Snapshot: {manifest.snapshot_id}")

    snapshot_dir = getattr(args, "snapshot_dir", "") or "doe_snapshots"
    from .snapshots import SnapshotManager
    snapshot_manager = SnapshotManager(snapshot_dir)

    record = replay_run(
        manifest=manifest,
        snapshot_manager=snapshot_manager,
    )

    if record is not None:
        print(f"\n  Replay complete: {record.run_id}")
        print(f"  State: {record.state.value}")
        if record.n_questions > 0:
            print(f"  Accuracy: {record.accuracy:.1%} "
                  f"({record.n_correct}/{record.n_questions})")
    else:
        print("  Replay returned no result.")


# ---------------------------------------------------------------------------
# Subcommand: regression promote / list
# ---------------------------------------------------------------------------

def _cmd_regression(args):
    """Manage regression test suites."""
    action = getattr(args, "regression_action", "list")

    suite_dir = getattr(args, "suite_dir", "") or "doe_regression"

    if action == "promote":
        run_id = getattr(args, "run_id", "")
        if not run_id:
            print("  ERROR: run_id required")
            return

        runs_dir = getattr(args, "runs_dir", "") or "doe_runs"
        from .runs import RunManager
        manager = RunManager(runs_dir)
        record = manager.get_run(run_id)
        if record is None:
            print(f"  Run not found: {run_id}")
            return
        if not record.result_path:
            print(f"  Run {run_id} has no result file")
            return

        from .replay import RegressionSuite

        suite = RegressionSuite(suite_dir)
        promoted = suite.promote_failures(record)
        if promoted:
            print(f"  Promoted {len(promoted)} failures to regression scenarios:")
            for sid in promoted:
                print(f"    {sid}")
        else:
            print("  No failures to promote.")

    elif action == "list":
        from .replay import RegressionSuite
        suite = RegressionSuite(suite_dir)
        scenario_ids = suite.list_scenarios()
        if not scenario_ids:
            print("  No regression scenarios.")
            return
        print(f"\n  Regression suite: {len(scenario_ids)} scenarios")
        for sid in scenario_ids:
            print(f"    {sid}")


# ---------------------------------------------------------------------------
# Subcommand: compare
# ---------------------------------------------------------------------------

def _cmd_compare(args):
    """Compare two benchmark results."""
    result_a_path = getattr(args, "result_a", "")
    result_b_path = getattr(args, "result_b", "")

    if not result_a_path or not result_b_path:
        print("  ERROR: two result paths required")
        return

    from knowlytix.harness.graphdoe.results import DOEBenchmarkResult
    from .replay import compare_runs

    result_a = DOEBenchmarkResult.load(result_a_path)
    result_b = DOEBenchmarkResult.load(result_b_path)

    comparison = compare_runs(result_a, result_b)

    print("\n  Run Comparison")
    print(f"  {'─' * 50}")
    print(f"  Run A: {comparison.run_a_id}")
    print(f"  Run B: {comparison.run_b_id}")
    print(f"  Accuracy delta: {comparison.accuracy_delta:+.1%}")
    print(f"  Geo confidence delta: {comparison.geo_confidence_delta:+.4f}")
    print(f"  Tension delta: {comparison.tension_delta:+.4f}")
    print(f"  Semantic drift: {comparison.semantic_drift:.4f}")
    print(f"  Regressions: {len(comparison.regressions)} "
          f"(correct -> wrong)")
    print(f"  Improvements: {len(comparison.improvements)} "
          f"(wrong -> correct)")
    print(f"  Stable pass: {comparison.stable_pass}")
    print(f"  Stable fail: {comparison.stable_fail}")

    if comparison.regressions:
        print("\n  Regressions:")
        for reg in comparison.regressions[:10]:
            qid = reg.get("doe_qid", "?")
            print(f"    {qid}")
        if len(comparison.regressions) > 10:
            print(f"    ... and {len(comparison.regressions) - 10} more")

    # Save comparison JSON if --output specified
    output = getattr(args, "output", "")
    if output:
        with open(output, "w") as f:
            json.dump(comparison.to_dict(), f, indent=2)
        print(f"\n  Comparison saved to {output}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="DOE-GMS Benchmark: systematic LLM/agent testing with "
                    "graph-verified ground truth and DOE factor analysis.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- Subcommand: run (also the default when no subcommand given) ---
    run_parser = subparsers.add_parser(
        "run", help="Run a DOE-GMS benchmark (default)")
    _add_run_arguments(run_parser)

    # --- Subcommand: runs ---
    runs_parser = subparsers.add_parser(
        "runs", help="Manage benchmark run records")
    runs_sub = runs_parser.add_subparsers(dest="runs_action")

    list_parser = runs_sub.add_parser("list", help="List all runs")
    list_parser.add_argument("--state", default="",
                             help="Filter by state")
    list_parser.add_argument("--runs-dir", default="doe_runs")

    show_parser = runs_sub.add_parser("show", help="Show run details")
    show_parser.add_argument("run_id", help="Run ID to show")
    show_parser.add_argument("--runs-dir", default="doe_runs")

    # --- Subcommand: replay ---
    replay_parser = subparsers.add_parser(
        "replay", help="Replay a benchmark from a manifest")
    replay_parser.add_argument(
        "manifest_path", help="Path to replay manifest YAML")
    replay_parser.add_argument("--snapshot-dir", default="doe_snapshots")

    # --- Subcommand: regression ---
    regression_parser = subparsers.add_parser(
        "regression", help="Manage regression test suites")
    regression_sub = regression_parser.add_subparsers(dest="regression_action")

    promote_parser = regression_sub.add_parser(
        "promote", help="Promote failures from a run to regression scenarios")
    promote_parser.add_argument("run_id", help="Source run ID")
    promote_parser.add_argument("--runs-dir", default="doe_runs")
    promote_parser.add_argument("--suite-dir", default="doe_regression")

    reg_list_parser = regression_sub.add_parser(
        "list", help="List regression scenarios")
    reg_list_parser.add_argument("--suite-dir", default="doe_regression")

    # --- Subcommand: compare ---
    compare_parser = subparsers.add_parser(
        "compare", help="Compare two benchmark results")
    compare_parser.add_argument("result_a", help="Path to result A JSON")
    compare_parser.add_argument("result_b", help="Path to result B JSON")
    compare_parser.add_argument("--output", default="",
                                help="Save comparison to JSON")

    # Parse
    args = parser.parse_args(argv)

    # Dispatch to subcommand
    command = args.command

    if command == "runs":
        return _cmd_runs(args)
    elif command == "replay":
        return _cmd_replay(args)
    elif command == "regression":
        return _cmd_regression(args)
    elif command == "compare":
        return _cmd_compare(args)
    elif command == "run":
        return _cmd_run(args)
    else:
        # No subcommand: treat as default run with positional markdown
        # Re-parse with the run parser to get all default arguments
        run_parser_fallback = argparse.ArgumentParser(
            description="DOE-GMS Benchmark")
        _add_run_arguments(run_parser_fallback)
        args = run_parser_fallback.parse_args(argv)
        return _cmd_run(args)


if __name__ == "__main__":
    main()
