"""End-to-end test of the DOE-GMS benchmark pipeline.

Runs: ingest → generate questions (classic + geometric) → GMS baseline → report.
Does NOT call any LLM — only tests the GMS side.
"""

import os

# Run from kg-memory root
os.chdir(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from knowlytix.harness.doe.config import DOEHarnessConfig
from knowlytix.harness.doe.benchmark import DOEGMSBenchmark


def main():
    md_path = os.path.join(
        os.path.dirname(__file__), "model_risk_assessment.md")

    config = DOEHarnessConfig(
        markdown_path=md_path,
        stores_dir="gms_stores",
        store_name="doe_test_model_risk",
        factor_group="quick_screen",
        n_runs=8,           # small for testing
        method="sobol",
        seed=42,
        max_per_category=5,
        enable_tracing=True,
        tracing_dir="doe_test_tracing",
        ingest_epochs=100,   # faster for testing
        device="cpu",        # avoid GPU memory issues
    )

    print("=" * 70)
    print("DOE-GMS Benchmark — End-to-End Test")
    print("=" * 70)

    # Step 1: Create benchmark
    print("\n[Step 1] Creating benchmark...")
    bench = DOEGMSBenchmark(config)

    # Step 2: Ingest markdown → GMS
    print("\n[Step 2] Ingesting markdown → GMS...")
    store = bench.ingest(md_path)
    stats = store.stats()
    print(f"  Store stats: {stats}")

    # Step 3: Check what the GMS learned
    print("\n[Step 3] GMS store contents:")
    if store.doc_graph:
        triples = list(store.doc_graph.triples)
        print(f"  Total triples: {len(triples)}")
        print("  Sample triples:")
        for h, r, t in triples[:10]:
            print(f"    ({h}, {r}, {t})")

    if store.enm:
        enm_keys = store.enm.keys()
        print(f"\n  Total ENM entries: {len(enm_keys)}")
        print("  Sample ENM:")
        for key in enm_keys[:10]:
            val = store.enm.read_exact(key)
            v = float(val.item() if hasattr(val, 'item') else val) if val is not None else None
            print(f"    [{key.type}] {key.id} = {v}")

    if store.adapter:
        print(f"\n  Entities: {len(store.adapter.entity_to_idx)}")
        print(f"  Relations: {len(store.adapter.relation_to_idx)}")

    # Step 4: Test geometric primitives
    print("\n[Step 4] Testing GMS geometric primitives...")

    # Score a triple
    if store.doc_graph and len(triples) > 0:
        h, r, t = triples[0]
        score = store.score_triple(h, r, t)
        print(f"  score_triple({h}, {r}, {t}) = {score}")

    # Tension energy
    entities = list(store.adapter.entity_to_idx.keys()) if store.adapter else []
    if len(entities) >= 2:
        te = store.tension_energy(entities[0], entities[1])
        print(f"  tension_energy({entities[0]}, {entities[1]}) = {te}")

    # Link prediction
    if store.doc_graph and len(triples) > 0:
        h, r, t = triples[0]
        preds = store.link_predict(h, r, top_k=3)
        print(f"  link_predict({h}, {r}, top_k=3) = {preds}")

    # Step 5: Generate questions (classic + geometric)
    print("\n[Step 5] Generating DOE questions...")
    doe_qs = bench.generate_questions()
    print(f"  Total DOE questions: {len(doe_qs)}")

    # Breakdown by category
    from collections import Counter
    cats = Counter(dq.question.category for dq in doe_qs)
    print("  By category:")
    for cat, count in sorted(cats.items()):
        print(f"    {cat}: {count}")

    # Step 6: Run GMS baseline
    print("\n[Step 6] Running GMS baseline (should be ~100%)...")
    baseline = bench.run_gms_baseline()
    bench.print_results(baseline)

    # Step 7: Save and report
    output_path = os.path.join(os.path.dirname(__file__), "baseline_results.json")
    baseline.save(output_path)
    print(f"\n  Results saved to {output_path}")

    bench.save_tracing()

    # Summary
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(f"  GMS store: {stats['entities']} entities, "
          f"{stats['triples']} triples, {stats['enm_entries']} ENM")
    print(f"  DOE questions: {len(doe_qs)} "
          f"({len(cats)} categories)")
    print(f"  GMS baseline: {baseline.total_correct}/{baseline.total_questions} "
          f"({baseline.accuracy:.0%})")

    return baseline


if __name__ == "__main__":
    main()
