# SPDX-License-Identifier: Apache-2.0
"""Train GMS on a FinStructBench instance and evaluate it.

Usage:
    python -m knowlytix.benchmark.train_gms finstructbench/instances/model_validation.md

Training and evaluation run in knowlytix-core
(``knowlytix.core.train_finstructbench``). This command line lives here because
it needs benchmark's ingest, question generators and scorer, and knowlytix-core
must not import knowlytix-benchmark.
"""

import argparse
import random

import numpy as np
import torch

from knowlytix.benchmark.generators import default_generators
from knowlytix.benchmark.ingest import ingest_markdown
from knowlytix.benchmark.scorers import score_answer
from knowlytix.core.diagnostics import GMSDiagnostics
from knowlytix.core.train_finstructbench import (
    GraphToGMS,
    evaluate_gms,
    populate_enm,
    print_link_prediction_samples,
    train_gms,
)


def main():
    parser = argparse.ArgumentParser(description="Train GMS on FinStructBench")
    parser.add_argument("markdown", help="Path to financial report markdown")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-neg", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--lr-riemannian", type=float, default=2e-3)
    parser.add_argument("--max-per-category", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-diagnostics", action="store_true",
                        help="Skip post-training diagnostics")
    args = parser.parse_args()

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Step 1: Ingest markdown -> DocumentGraph
    print(f"Ingesting: {args.markdown}")
    doc_graph = ingest_markdown(args.markdown)
    stats = doc_graph.stats()
    print(f"  ENM entries: {stats['enm_entries']}")
    print(f"  KG triples:  {stats['triples']}")

    # Step 2: Convert to GMS training data
    print("\nConverting to GMS format...")
    adapter = GraphToGMS(doc_graph)

    # Step 3: Train GMS (uses LossConfig defaults — validated hyperparameters)
    model = train_gms(
        adapter, device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_neg=args.num_neg,
        lr=args.lr,
        lr_riemannian=args.lr_riemannian,
    )

    # Step 4: Populate ENM
    print("\nPopulating ENM register...")
    enm = populate_enm(doc_graph, adapter)

    # Step 5: Post-training diagnostics
    if not args.no_diagnostics:
        diag = GMSDiagnostics(model, adapter, enm, doc_graph, device)
        report = diag.run_all()
        GMSDiagnostics.print_report(report)

    # Step 6: Evaluate
    evaluate_gms(
        model, adapter, enm, doc_graph, device,
        generators=default_generators(),
        score_answer=score_answer,
        max_per_category=args.max_per_category,
        seed=args.seed,
    )

    # Step 7: Link prediction sample
    print_link_prediction_samples(model, adapter, enm, doc_graph, device)


if __name__ == "__main__":
    main()
