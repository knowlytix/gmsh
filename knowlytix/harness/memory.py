# SPDX-License-Identifier: Apache-2.0
"""GMS memory integration: extract triples from MoDeva results, learn, recall."""

import torch

from knowlytix.knowledge.config import DocGMSConfig
from knowlytix.knowledge.extract import ExtractedTriple
from knowlytix.knowledge.learn import RuntimeLearner
from knowlytix.knowledge.store import GMSExpertStore


# ── Triple Extraction ──────────────────────────────────────────

def extract_triples_from_result(result, dataset_name: str,
                                 model_name: str | None = None) -> list[ExtractedTriple]:
    """Deterministic triple extraction from MoDeva ValidationResult.

    No LLM calls — uses rule-based pattern matching on result.key and result.table.
    """
    if result is None:
        return []

    triples = []
    key = getattr(result, "key", "") or ""

    if "accuracy" in key and "compare" not in key:
        triples.extend(_extract_accuracy(result, model_name))
    elif "compare_accuracy" in key:
        triples.extend(_extract_compare_accuracy(result, dataset_name))
    elif "robustness" in key:
        triples.extend(_extract_robustness(result, model_name))
    elif "reliability" in key:
        triples.extend(_extract_reliability(result, model_name))
    elif "resilience" in key:
        triples.extend(_extract_resilience(result, model_name))
    elif "fairness" in key:
        triples.extend(_extract_fairness(result, model_name))
    elif "weakness" in key:
        triples.extend(_extract_weakness(result, model_name))
    elif "pfi" in key or "interpret_fi" in key or "interpret_ei" in key:
        triples.extend(_extract_feature_importance(result, model_name))
    elif "summary" in key:
        triples.extend(_extract_data_summary(result, dataset_name))

    return triples


def _extract_accuracy(result, model_name):
    """Extract metric triples from accuracy table."""
    triples = []
    if not model_name:
        return triples

    tbl = getattr(result, "table", None)
    val = getattr(result, "value", None)
    if val and isinstance(val, dict):
        for metric, scores in val.items():
            if isinstance(scores, dict):
                for split, score in scores.items():
                    if split == "GAP":
                        triples.append(ExtractedTriple(
                            head=model_name,
                            relation=f"train_test_gap_{metric}",
                            tail=str(score),
                            numeric_value=float(score) if _is_numeric(score) else None,
                        ))
                    else:
                        triples.append(ExtractedTriple(
                            head=model_name,
                            relation=f"has_{split}_{metric}",
                            tail=str(score),
                            numeric_value=float(score) if _is_numeric(score) else None,
                        ))
    return triples


def _extract_compare_accuracy(result, dataset_name):
    """Extract comparison triples from multi-model accuracy."""
    triples = []
    val = getattr(result, "value", None)
    if not val or not isinstance(val, dict):
        return triples

    # Find best model per metric
    model_scores = {}
    for model_name, metrics in val.items():
        if isinstance(metrics, dict):
            for metric, scores in metrics.items():
                if isinstance(scores, dict) and "test" in scores:
                    test_score = scores.get("test", scores.get(list(scores.keys())[0]))
                    if _is_numeric(test_score):
                        model_scores.setdefault(metric, []).append(
                            (model_name, float(test_score))
                        )
                        triples.append(ExtractedTriple(
                            head=model_name,
                            relation=f"has_test_{metric}",
                            tail=str(test_score),
                            numeric_value=float(test_score),
                        ))

    # Determine best model per metric
    for metric, scores in model_scores.items():
        if scores:
            best = max(scores, key=lambda x: x[1])
            triples.append(ExtractedTriple(
                head=dataset_name or "dataset",
                relation=f"best_model_by_{metric}",
                tail=best[0],
            ))

    return triples


def _extract_robustness(result, model_name):
    triples = []
    if not model_name:
        return triples

    tbl = getattr(result, "table", None)
    if tbl is not None and hasattr(tbl, "mean"):
        # Table rows are repeats, columns are noise levels + original
        means = tbl.mean()
        for col in means.index:
            triples.append(ExtractedTriple(
                head=model_name,
                relation=f"robustness_at_{col}",
                tail=str(round(float(means[col]), 4)),
                numeric_value=float(means[col]),
            ))
    return triples


def _extract_reliability(result, model_name):
    triples = []
    if not model_name:
        return triples

    tbl = getattr(result, "table", None)
    if tbl is not None and hasattr(tbl, "to_dict"):
        for col in tbl.columns:
            for idx in tbl.index:
                val = tbl.loc[idx, col]
                if _is_numeric(val):
                    triples.append(ExtractedTriple(
                        head=model_name,
                        relation=f"reliability_{col}",
                        tail=str(val),
                        numeric_value=float(val),
                    ))
    return triples


def _extract_resilience(result, model_name):
    triples = []
    if not model_name:
        return triples

    tbl = getattr(result, "table", None)
    if tbl is not None and hasattr(tbl, "iloc"):
        # Last row is full dataset, first rows are worst subsets
        if len(tbl) > 0:
            worst_row = tbl.iloc[0]
            for col in tbl.columns:
                if _is_numeric(worst_row[col]):
                    triples.append(ExtractedTriple(
                        head=model_name,
                        relation=f"resilience_worst_{col}",
                        tail=str(worst_row[col]),
                        numeric_value=float(worst_row[col]),
                    ))
    return triples


def _extract_fairness(result, model_name):
    triples = []
    if not model_name:
        return triples

    val = getattr(result, "value", None)
    if val and isinstance(val, dict):
        for group, info in val.items():
            if isinstance(info, dict):
                fm = info.get("fairness_metric")
                if _is_numeric(fm):
                    triples.append(ExtractedTriple(
                        head=model_name,
                        relation=f"fairness_{group}",
                        tail=str(fm),
                        numeric_value=float(fm),
                    ))
    return triples


def _extract_weakness(result, model_name):
    triples = []
    if not model_name:
        return triples

    val = getattr(result, "value", None)
    if val and isinstance(val, dict):
        # Extract JS divergence ranking
        js_rank = val.get("js_divergence_ranking")
        if js_rank is not None and hasattr(js_rank, "__len__"):
            if hasattr(js_rank, "items"):
                for feat, score in list(js_rank.items())[:5]:
                    triples.append(ExtractedTriple(
                        head=model_name,
                        relation="has_weakness_in",
                        tail=str(feat),
                        numeric_value=float(score) if _is_numeric(score) else None,
                    ))
            elif hasattr(js_rank, "__iter__"):
                for item in list(js_rank)[:5]:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        triples.append(ExtractedTriple(
                            head=model_name,
                            relation="has_weakness_in",
                            tail=str(item[0]),
                            numeric_value=float(item[1]) if _is_numeric(item[1]) else None,
                        ))

    tbl = getattr(result, "table", None)
    if tbl is not None and hasattr(tbl, "shape"):
        n_weak = tbl[tbl.get("Is Weak", tbl.iloc[:, -1]).astype(bool)].shape[0] if "Is Weak" in tbl.columns else 0
        if n_weak > 0:
            triples.append(ExtractedTriple(
                head=model_name,
                relation="n_weak_regions",
                tail=str(n_weak),
                numeric_value=float(n_weak),
            ))
    return triples


def _extract_feature_importance(result, model_name):
    triples = []
    if not model_name:
        return triples

    tbl = getattr(result, "table", None)
    if tbl is not None and hasattr(tbl, "iterrows"):
        for _, row in tbl.iterrows():
            name_col = "Name" if "Name" in tbl.columns else tbl.columns[0]
            score_col = next(
                (c for c in ["Score", "Importance", "Effect"] if c in tbl.columns),
                tbl.columns[1] if len(tbl.columns) > 1 else None,
            )
            if score_col:
                feat_name = row[name_col]
                feat_score = row[score_col]
                if _is_numeric(feat_score):
                    triples.append(ExtractedTriple(
                        head=model_name,
                        relation="feature_importance",
                        tail=str(feat_name),
                        numeric_value=float(feat_score),
                    ))
    return triples


def _extract_data_summary(result, dataset_name):
    triples = []
    if not dataset_name:
        return triples

    tbl = getattr(result, "table", None)
    if tbl and isinstance(tbl, dict):
        summary = tbl.get("summary")
        if summary is not None and hasattr(summary, "shape"):
            n_samples = summary.shape[0] if hasattr(summary, "shape") else 0
            triples.append(ExtractedTriple(
                head=dataset_name,
                relation="has_features",
                tail=str(summary.shape[1] if len(summary.shape) > 1 else 0),
            ))

    val = getattr(result, "value", None)
    if val and isinstance(val, dict):
        summary_info = val.get("summary", {})
        if isinstance(summary_info, dict):
            for k, v in summary_info.items():
                if _is_numeric(v):
                    triples.append(ExtractedTriple(
                        head=dataset_name,
                        relation=f"summary_{k}",
                        tail=str(v),
                        numeric_value=float(v),
                    ))
    return triples


def _is_numeric(val):
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False


# ── Memory Context Gathering ───────────────────────────────────

def gather_memory_context(modeva_store: GMSExpertStore | None,
                           session_store: GMSExpertStore | None,
                           dataset_name: str | None = None,
                           model_names: list[str] | None = None,
                           question: str = "") -> tuple[str, str]:
    """Gather context from both GMS stores.

    Returns (modeva_context, session_context).
    """
    modeva_ctx = ""
    session_ctx = ""

    # Query MoDeva GMS for methodology knowledge
    if modeva_store is not None and modeva_store.doc_graph is not None:
        modeva_ctx = _query_store_context(modeva_store, question,
                                           entities_to_check=None,
                                           max_lines=30)

    # Query Session GMS for experiment-specific knowledge
    if session_store is not None and session_store.doc_graph is not None:
        entities = []
        if dataset_name:
            entities.append(dataset_name)
        if model_names:
            entities.extend(model_names)
        session_ctx = _query_store_context(session_store, question,
                                            entities_to_check=entities,
                                            max_lines=30)

    return modeva_ctx, session_ctx


def _query_store_context(store: GMSExpertStore, question: str,
                          entities_to_check: list[str] | None,
                          max_lines: int = 30) -> str:
    """Query a GMS store for relevant triples."""
    lines = []

    # Query triples for known entities
    if entities_to_check:
        for entity in entities_to_check:
            matched = store.fuzzy_match_entity(entity)
            if matched:
                triples = store.query_triples(head=matched)
                for h, r, t in triples[:10]:
                    lines.append(f"[{h}] {r} → {t}")

                # Check ENM values
                if store.enm is not None:
                    for cat in ["metric", "score", "value"]:
                        val = store.lookup_enm(cat, matched)
                        if val is not None:
                            lines.append(f"[{matched}] ENM({cat}) = {val}")

    # Also try matching words from the question
    if question:
        words = [w.lower().strip() for w in question.split()
                 if len(w) > 3 and w.lower() not in {
                     "what", "which", "where", "when", "does", "have",
                     "that", "this", "with", "from", "about", "the",
                 }]
        for word in words[:5]:
            matched = store.fuzzy_match_entity(word)
            if matched and not any(matched in line for line in lines):
                triples = store.query_triples(head=matched)
                for h, r, t in triples[:5]:
                    lines.append(f"[{h}] {r} → {t}")

    return "\n".join(lines[:max_lines])


# ── Learning ───────────────────────────────────────────────────

def learn_from_result(learner: RuntimeLearner | None,
                       result, dataset_name: str,
                       model_name: str | None,
                       device: torch.device):
    """Extract triples from result and learn into session GMS."""
    if learner is None or result is None:
        return None

    triples = extract_triples_from_result(result, dataset_name, model_name)
    if not triples:
        return None

    try:
        learn_result = learner.learn_triples(triples, device)
        return learn_result
    except Exception as e:
        print(f"  [Memory] Learning error: {e}")
        return None


# ── Store Bootstrap ────────────────────────────────────────────

def bootstrap_session_store(store: GMSExpertStore, config: DocGMSConfig,
                             device: torch.device):
    """Initialize a minimal session GMS for the first run.

    Creates a seed graph with common relation types so the store can
    accept new triples incrementally.
    """
    from knowlytix.benchmark.graph import DocumentGraph
    from knowlytix.core.train_finstructbench import GraphToGMS, train_gms, populate_enm

    doc_graph = DocumentGraph()

    # Seed entities and relations
    seeds = [
        ("data_science_agent", "is_type", "agent"),
        ("modeva", "is_type", "framework"),
        ("dataset", "has_property", "task_type"),
        ("model", "has_property", "accuracy"),
        ("model", "has_property", "robustness"),
        ("model", "has_property", "fairness"),
        ("model", "has_property", "reliability"),
        ("model", "has_property", "resilience"),
        ("fanova", "supports", "interpretability"),
        ("depth_2_trees", "enables", "fanova"),
        ("gaminet", "is_type", "interpretable_model"),
        ("directrs", "is_type", "interpretable_model"),
        ("xgboost", "is_type", "tree_ensemble"),
        ("lightgbm", "is_type", "tree_ensemble"),
        ("weakness_region", "diagnoses", "failure_zones"),
        ("amif", "detects", "weakness_region"),
        ("pfi", "measures", "feature_importance"),
        ("pdp", "shows", "marginal_effect"),
        ("shap", "explains", "local_prediction"),
        ("conformal_prediction", "measures", "reliability"),
        ("tension_energy", "detects", "contradiction"),
    ]

    for h, r, t in seeds:
        doc_graph.add_triple(h, r, t)

    store.doc_graph = doc_graph

    # Build adapter + train minimal GMS
    adapter = GraphToGMS(doc_graph)
    model = train_gms(
        adapter, device,
        epochs=config.train.epochs,
        batch_size=config.train.batch_size,
    )
    enm = populate_enm(doc_graph, adapter)

    from knowlytix.core.graph.transport import RelationalTransport
    transport = RelationalTransport(model)

    store.model = model
    store.adapter = adapter
    store.enm = enm
    store.transport = transport
    store.markdown = "Session knowledge store for Data Science Agent."
    store.save()
    print(f"  [Memory] Session store bootstrapped with {len(seeds)} seed triples.")
