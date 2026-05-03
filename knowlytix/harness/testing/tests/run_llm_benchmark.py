"""Run DOE-GMS benchmark against Sonnet and Opus 4.6.

Evaluates both models on the same DOE questions with geometric judging,
then prints a comparative analysis.
"""

import os
import time
import subprocess

os.chdir(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Best-effort: load ANTHROPIC_API_KEY from the local `pass` store when
# absent so developers on the dev box don't have to export it manually.
# Absence here is no longer a hard error — the benchmark surfaces a
# `gms.llm.LLMConfigError` at the first LLM call with an actionable
# message (env var name + override hints) if no key is available.
if not os.environ.get("ANTHROPIC_API_KEY"):
    try:
        key = subprocess.check_output(
            ["pass", "show", "anthropic/api-key"],
            text=True).strip()
        os.environ["ANTHROPIC_API_KEY"] = key
    except Exception:
        # Let gms.llm.LLMConfigError surface when/if a call is made —
        # it names the correct env var for whichever provider the user
        # configures (GMS_LLM_MODEL may point at Ollama, OpenAI, etc.).
        pass

from knowlytix.harness.testing.config import DOEHarnessConfig  # noqa: E402
from knowlytix.harness.testing.benchmark import DOEGMSBenchmark  # noqa: E402
from knowlytix.harness.testing.evaluators import make_evaluator  # noqa: E402


MODELS = [
    "claude-sonnet-4-20250514",
    "claude-opus-4-7",
]


def run_model(bench, model_name, output_dir):
    """Run benchmark against a single model."""
    print(f"\n{'#' * 70}")
    print(f"# EVALUATING: {model_name}")
    print(f"{'#' * 70}")

    evaluator = make_evaluator("llm", model_name)
    result = bench.run(evaluator=evaluator, model=model_name)

    # Print results
    bench.print_results(result)

    # Save
    out_path = os.path.join(output_dir, f"doe_{model_name.replace('-', '_')}.json")
    result.save(out_path)
    print(f"\n  Saved to {out_path}")

    # Analyze
    try:
        analyzer = bench.analyze(result)
        analyzer.print_report()
    except Exception as e:
        print(f"\n  Analysis error: {e}")

    return result


def compare_results(results: dict):
    """Print comparative summary across models."""
    print(f"\n{'=' * 70}")
    print("COMPARATIVE SUMMARY")
    print(f"{'=' * 70}")

    # Overall accuracy
    print(f"\n  {'Model':<35} {'Correct':>8} {'Total':>6} {'Accuracy':>9}")
    print(f"  {'─' * 60}")
    for model, result in results.items():
        short = model.split("/")[-1] if "/" in model else model
        print(f"  {short:<35} {result.total_correct:>8} "
              f"{result.total_questions:>6} "
              f"{result.accuracy:>8.1%}")

    # Per-category comparison
    print("\n  Per-Category Accuracy:")
    categories = set()
    for result in results.values():
        df = result.to_dataframe()
        if not df.empty:
            categories.update(df["category"].unique())

    if categories:
        header = f"  {'Category':<22}"
        for model in results:
            short = model.replace("claude-", "").replace("-20250514", "")
            header += f" {short:>12}"
        print(header)
        print(f"  {'─' * (22 + 13 * len(results))}")

        for cat in sorted(categories):
            line = f"  {cat:<22}"
            for model, result in results.items():
                df = result.to_dataframe()
                if not df.empty and cat in df["category"].values:
                    sub = df[df["category"] == cat]
                    rate = sub["correct"].mean()
                    line += f" {rate:>11.0%}"
                else:
                    line += f" {'N/A':>12}"
            print(line)

    # Geometric scores comparison
    print("\n  Geometric Scores:")
    geo_header = f"  {'Metric':<25}"
    for model in results:
        short = model.replace("claude-", "").replace("-20250514", "")
        geo_header += f" {short:>12}"
    print(geo_header)
    print(f"  {'─' * (25 + 13 * len(results))}")

    for metric_name, col in [
        ("Geodesic Confidence", "_geo_confidence"),
        ("Tension Score", "_tension"),
        ("Holonomy Score", "_holonomy"),
        ("Overall Confidence", "_overall_confidence"),
    ]:
        line = f"  {metric_name:<25}"
        for model, result in results.items():
            df = result.to_dataframe()
            if not df.empty and col in df.columns:
                val = df[col].mean()
                line += f" {val:>11.3f}"
            else:
                line += f" {'N/A':>12}"
        print(line)


def main():
    md_path = os.path.join(
        os.path.dirname(__file__), "model_risk_assessment.md")
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)

    config = DOEHarnessConfig(
        markdown_path=md_path,
        stores_dir="gms_stores",
        store_name="doe_test_model_risk",
        factor_group="quick_screen",
        n_runs=8,
        method="sobol",
        seed=42,
        max_per_category=5,
        enable_tracing=True,
        tracing_dir="doe_test_tracing",
        device="cpu",
    )

    print("=" * 70)
    print("DOE-GMS Benchmark — Sonnet vs Opus 4.6")
    print("=" * 70)

    bench = DOEGMSBenchmark(config)
    bench.ingest(md_path)

    # Generate questions once (shared across models)
    bench.generate_questions()

    # Run each model
    results = {}
    for model in MODELS:
        t0 = time.time()
        # Reset questions so run() reuses the cached ones
        result = run_model(bench, model, output_dir)
        elapsed = time.time() - t0
        results[model] = result
        print(f"\n  {model} completed in {elapsed:.0f}s")

    # Compare
    compare_results(results)

    # Save tracing
    bench.save_tracing()

    print(f"\n{'=' * 70}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
