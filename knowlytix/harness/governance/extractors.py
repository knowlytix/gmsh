# SPDX-License-Identifier: Apache-2.0
"""Three-tier knowledge extraction from tool outputs.

Tier 1: Deterministic parsing of raw dicts (100% accuracy).
Tier 2: Regex-based extraction from display strings (fallback).
Tier 3: LLM-based extraction via knowlytix.knowledge.extract (most flexible, noisiest).

Reuses knowlytix.knowledge.extract infrastructure:
  - ExtractedTriple for the triple data class
  - extract_triples() for Tier 3 LLM extraction
  - score_triples() / assess_plausibility() for GMS verification

Built in Chapter 4 of "How to Build Policy-Governed AI Agents."
"""

import re
from dataclasses import dataclass
from typing import Any

from knowlytix.knowledge.extract import (
    ExtractedTriple,
    extract_triples_simple,
)


# ── Data Classes ──────────────────────────────────────────────────

@dataclass
class ENMEntry:
    """An Entity-Numeric-Map entry for phase-encoded storage."""
    category: str       # e.g. "metric", "robustness", "fairness"
    entity_id: str      # e.g. "rf_100_test_accuracy"
    value: float        # the numeric value


@dataclass
class ExtractionResult:
    """Unified output from any extraction tier."""
    triples: list[ExtractedTriple]    # reuse from knowlytix.knowledge.extract
    enm_entries: list[ENMEntry]       # numeric facts for phase encoding
    tier: int                         # 1, 2, or 3
    tool_name: str = ""               # which tool produced this
    n_facts: int = 0                  # convenience: len(triples) + len(enm_entries)

    def __post_init__(self):
        self.n_facts = len(self.triples) + len(self.enm_entries)


# ── Tier 1: Deterministic Extractors ─────────────────────────────
# Each function parses a raw dict returned by governed_agent.tools
# and produces triples + ENM entries with zero ambiguity.

def _as_float(val) -> float | None:
    """Best-effort numeric cast; returns None for values that cannot
    round-trip through float(). Used to keep the Tier 1 extractors
    robust against tools that return None, NaN or non-numeric strings
    for a metric slot."""
    if val is None:
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN check
        return None
    return out


def _tier1_load_dataset(args: dict, raw: dict) -> ExtractionResult:
    """Extract from load_dataset raw dict."""
    name = raw.get("name", args.get("name", "unknown"))
    task_type = raw.get("task_type", "unknown")
    target_name = raw.get("target_name", "unknown")
    n_features = _as_float(raw.get("n_features"))
    n_samples = _as_float(raw.get("n_samples"))
    train_size = _as_float(raw.get("train_size"))
    test_size = _as_float(raw.get("test_size"))

    triples = [
        ExtractedTriple(name, "is_type", "dataset"),
        ExtractedTriple(name, "has_task_type", task_type),
        ExtractedTriple(name, "has_target", target_name),
    ]
    enm = []
    if n_features is not None:
        triples.append(ExtractedTriple(
            name, "has_n_features", str(int(n_features)),
            numeric_value=n_features,
        ))
        enm.append(ENMEntry("data", f"{name}_n_features", n_features))
    if n_samples is not None:
        triples.append(ExtractedTriple(
            name, "has_n_samples", str(int(n_samples)),
            numeric_value=n_samples,
        ))
        enm.append(ENMEntry("data", f"{name}_n_samples", n_samples))
    if train_size is not None:
        enm.append(ENMEntry("data", f"{name}_train_size", train_size))
    if test_size is not None:
        enm.append(ENMEntry("data", f"{name}_test_size", test_size))

    # Feature-name triples
    for feat in raw.get("feature_names", [])[:10]:  # cap to avoid explosion
        triples.append(ExtractedTriple(name, "has_feature", feat))

    return ExtractionResult(triples=triples, enm_entries=enm, tier=1,
                            tool_name="load_dataset")


def _tier1_train_model(args: dict, raw: dict) -> ExtractionResult:
    """Extract from train_model raw dict."""
    model_name = raw.get("model_name", args.get("model_name", "unknown"))
    model_type = raw.get("model_type", args.get("model_type", "unknown"))
    metrics = raw.get("metrics", {})

    triples = [
        ExtractedTriple(model_name, "is_type", model_type),
        ExtractedTriple(model_name, "trained_on", args.get("dataset", "unknown")),
    ]
    enm = []

    for metric_key, metric_val in metrics.items():
        v = _as_float(metric_val)
        if v is None:
            continue
        triples.append(ExtractedTriple(
            model_name, f"has_{metric_key}", str(v),
            numeric_value=v,
        ))
        enm.append(ENMEntry("metric", f"{model_name}_{metric_key}", v))

    return ExtractionResult(triples=triples, enm_entries=enm, tier=1,
                            tool_name="train_model")


def _tier1_diagnose_robustness(args: dict, raw: dict) -> ExtractionResult:
    """Extract from diagnose_robustness raw dict."""
    model_name = raw.get("model_name", args.get("model_name", "unknown"))
    noise_results = raw.get("noise_results", {})

    triples = [
        ExtractedTriple(model_name, "diagnosed_for", "robustness"),
    ]
    enm = []

    for noise_key, noise_vals in noise_results.items():
        for metric_name, metric_val in noise_vals.items():
            v = _as_float(metric_val)
            if v is None:
                continue
            triple_rel = f"robustness_{noise_key}_{metric_name}"
            triples.append(ExtractedTriple(
                model_name, triple_rel, str(v),
                numeric_value=v,
            ))
            enm.append(ENMEntry(
                "robustness",
                f"{model_name}_{noise_key}_{metric_name}",
                v,
            ))

    # Compute degradation assessment
    def _noise_key(k: str) -> float:
        try:
            return float(k.split("_")[1]) if "_" in k else 0.0
        except (ValueError, IndexError):
            return 0.0
    sorted_keys = sorted(noise_results.keys(), key=_noise_key)
    if len(sorted_keys) >= 2:
        first = noise_results[sorted_keys[0]] or {}
        last = noise_results[sorted_keys[-1]] or {}
        first_val = next(iter(first.values()), None) if first else None
        last_val = next(iter(last.values()), None) if last else None
        first_f = _as_float(first_val)
        last_f = _as_float(last_val)
        if first_f is not None and last_f is not None:
            drop = round(first_f - last_f, 4)
            severity = "low" if abs(drop) < 0.05 else (
                "medium" if abs(drop) < 0.15 else "high")
            triples.append(ExtractedTriple(
                model_name, "robustness_degradation", str(drop),
                numeric_value=drop,
            ))
            triples.append(ExtractedTriple(
                model_name, "robustness_severity", severity,
            ))

    return ExtractionResult(triples=triples, enm_entries=enm, tier=1,
                            tool_name="diagnose_robustness")


def _tier1_diagnose_fairness(args: dict, raw: dict) -> ExtractionResult:
    """Extract from diagnose_fairness raw dict."""
    model_name = args.get("model_name", "unknown")
    protected = raw.get("protected_attribute", "unknown")

    triples = [
        ExtractedTriple(model_name, "diagnosed_for", "fairness"),
        ExtractedTriple(model_name, "protected_attribute", protected),
    ]
    enm = []

    air = _as_float(raw.get("adverse_impact_ratio"))
    if air is not None:
        triples.append(ExtractedTriple(
            model_name, "has_air", str(air), numeric_value=air,
        ))
        enm.append(ENMEntry("fairness", f"{model_name}_air", air))

    dp = _as_float(raw.get("demographic_parity_diff"))
    if dp is not None:
        triples.append(ExtractedTriple(
            model_name, "has_dp_diff", str(dp), numeric_value=dp,
        ))
        enm.append(ENMEntry("fairness", f"{model_name}_dp_diff", dp))

    passes = raw.get("air_passes_4_5_rule")
    if passes is not None:
        triples.append(ExtractedTriple(
            model_name, "passes_4_5_rule", str(passes),
        ))

    # Per-group rates
    for group, rate in raw.get("group_positive_rates", {}).items():
        r = _as_float(rate)
        if r is None:
            continue
        triples.append(ExtractedTriple(
            model_name, f"selection_rate_{group}", str(r),
            numeric_value=r,
        ))
        enm.append(ENMEntry(
            "fairness", f"{model_name}_sel_rate_{group}", r,
        ))

    return ExtractionResult(triples=triples, enm_entries=enm, tier=1,
                            tool_name="diagnose_fairness")


def _tier1_explain_importance(args: dict, raw: dict) -> ExtractionResult:
    """Extract from explain_importance raw dict."""
    model_name = raw.get("model_name", args.get("model_name", "unknown"))
    importances = raw.get("importances", {})

    triples = [
        ExtractedTriple(model_name, "diagnosed_for", "feature_importance"),
    ]
    enm = []

    numeric_importances = {}
    for feat, imp in importances.items():
        v = _as_float(imp)
        if v is None:
            continue
        numeric_importances[feat] = v
        triples.append(ExtractedTriple(
            model_name, f"importance_of_{feat}", str(v),
            numeric_value=v,
        ))
        enm.append(ENMEntry(
            "importance", f"{model_name}_pfi_{feat}", v,
        ))

    # Top driver
    if numeric_importances:
        top_feat = max(numeric_importances, key=numeric_importances.get)
        triples.append(ExtractedTriple(
            top_feat, "is_top_driver_for", model_name,
        ))

    return ExtractionResult(triples=triples, enm_entries=enm, tier=1,
                            tool_name="explain_importance")


def _tier1_leaderboard(args: dict, raw: dict) -> ExtractionResult:
    """Extract from leaderboard raw dict."""
    ranking = raw.get("ranking", [])
    sort_by = raw.get("sort_by", "metric")

    triples = []
    enm = []

    for entry in ranking:
        mname = entry.get("model", "unknown")
        mtype = entry.get("type", "unknown")
        rank_f = _as_float(entry.get("rank", 0)) or 0.0

        triples.append(ExtractedTriple(
            mname, "is_type", mtype,
        ))
        triples.append(ExtractedTriple(
            mname, "has_leaderboard_rank", str(int(rank_f)),
            numeric_value=rank_f,
        ))
        enm.append(ENMEntry("ranking", f"{mname}_rank", rank_f))

        # Extract all metrics from the nested metrics dict
        for mk, mv in entry.get("metrics", {}).items():
            v = _as_float(mv)
            if v is None:
                continue
            triples.append(ExtractedTriple(
                mname, f"has_{mk}", str(v),
                numeric_value=v,
            ))
            enm.append(ENMEntry("metric", f"{mname}_{mk}", v))

    # Who is rank 1?
    if ranking:
        best = ranking[0]
        triples.append(ExtractedTriple(
            best.get("model", "unknown"), "is_best_model_by", sort_by,
        ))

    return ExtractionResult(triples=triples, enm_entries=enm, tier=1,
                            tool_name="leaderboard")


# ── Tier 1 dispatcher ────────────────────────────────────────────

_TIER1_EXTRACTORS = {
    "load_dataset":          _tier1_load_dataset,
    "train_model":           _tier1_train_model,
    "diagnose_robustness":   _tier1_diagnose_robustness,
    "diagnose_fairness":     _tier1_diagnose_fairness,
    "explain_importance":    _tier1_explain_importance,
    "leaderboard":           _tier1_leaderboard,
}


def tier1_extract(tool_name: str, args: dict,
                  raw_result: dict) -> ExtractionResult | None:
    """Attempt Tier 1 deterministic extraction.

    Returns None if no Tier 1 extractor exists for this tool.
    """
    extractor = _TIER1_EXTRACTORS.get(tool_name)
    if extractor is None:
        return None
    if raw_result is None or "error" in raw_result:
        return None
    return extractor(args, raw_result)


# ── Tier 2: Regex-Based Extraction from Display Strings ──────────

_METRIC_RE = re.compile(
    r"(\w[\w\s]*?):\s*(-?\d+\.?\d*(?:e[+-]?\d+)?)", re.IGNORECASE
)

_LEADERBOARD_ROW_RE = re.compile(
    r"(\d+)\s+(\S+)\s+(\S+)\s+(-?\d+\.?\d*)"
)


def tier2_extract(tool_name: str, args: dict,
                  display_string: str) -> ExtractionResult:
    """Regex-based extraction from the human-readable display string.

    Less reliable than Tier 1 because display strings are formatted
    for human consumption, not machine parsing.
    """
    triples = []
    enm = []
    context = args.get("model_name", args.get("name",
              args.get("dataset", "unknown")))

    if tool_name == "leaderboard":
        # Try to parse the leaderboard table
        for m in _LEADERBOARD_ROW_RE.finditer(display_string):
            rank, model, mtype, val = m.groups()
            triples.append(ExtractedTriple(
                model, "has_leaderboard_rank", rank,
                numeric_value=float(rank),
            ))
            triples.append(ExtractedTriple(
                model, "is_type", mtype,
            ))
            enm.append(ENMEntry("ranking", f"{model}_rank", float(rank)))
    else:
        # Generic: extract all key: value pairs
        for m in _METRIC_RE.finditer(display_string):
            key = m.group(1).strip().replace(" ", "_").lower()
            val = float(m.group(2))
            triples.append(ExtractedTriple(
                context, f"has_{key}", str(val),
                numeric_value=val,
            ))
            enm.append(ENMEntry("metric", f"{context}_{key}", val))

    return ExtractionResult(triples=triples, enm_entries=enm, tier=2,
                            tool_name=tool_name)


# ── Tier 3: LLM / Simple-Parse Extraction ────────────────────────

def tier3_extract(tool_name: str, args: dict,
                  display_string: str) -> ExtractionResult:
    """Tier 3: parse TRIPLE: lines from display text or LLM output.

    In production this would call knowlytix.knowledge.extract.extract_triples() with
    an LLM backend.  For deterministic operation (no LLM key required)
    we use extract_triples_simple() which parses structured text.
    """
    triples = extract_triples_simple(display_string)

    # Also try the generic metric regex as a supplement
    enm = []
    context = args.get("model_name", args.get("name", "unknown"))
    for t in triples:
        if t.numeric_value is not None:
            enm.append(ENMEntry(
                "extracted", f"{t.head}_{t.relation}", t.numeric_value,
            ))

    return ExtractionResult(triples=triples, enm_entries=enm, tier=3,
                            tool_name=tool_name)


# ── Unified Dispatcher ────────────────────────────────────────────

def extract_from_tool(tool_name: str, args: dict,
                      raw_result: Any,
                      display_string: str = "") -> ExtractionResult:
    """Unified three-tier extraction dispatcher.

    Tries Tier 1 first (deterministic, 100% accuracy).
    Falls back to Tier 2 (regex on display string).
    Falls back to Tier 3 (structured text parser / LLM).

    Parameters
    ----------
    tool_name : str
        Name of the tool that produced the result.
    args : dict
        Arguments originally passed to the tool.
    raw_result : dict or None
        The raw Python dict returned alongside the display string.
    display_string : str
        The human-readable display string from the tool.

    Returns
    -------
    ExtractionResult
        Triples, ENM entries and the tier that produced them.
    """
    # Tier 1: deterministic parsing of raw dict
    if raw_result is not None and isinstance(raw_result, dict):
        result = tier1_extract(tool_name, args, raw_result)
        if result is not None and result.n_facts > 0:
            return result

    # Tier 2: regex on display string
    if display_string:
        result = tier2_extract(tool_name, args, display_string)
        if result.n_facts > 0:
            return result

    # Tier 3: structured text parser (or LLM in production)
    if display_string:
        return tier3_extract(tool_name, args, display_string)

    # Nothing to extract
    return ExtractionResult(triples=[], enm_entries=[], tier=0,
                            tool_name=tool_name)


# ── Accuracy Comparison Helper ────────────────────────────────────

def compare_extraction_accuracy(
    ground_truth: dict[str, float],
    extraction: ExtractionResult,
) -> dict:
    """Compare extracted numeric values against known ground truth.

    Parameters
    ----------
    ground_truth : dict
        Mapping of metric name -> expected value.
    extraction : ExtractionResult
        The extraction result to evaluate.

    Returns
    -------
    dict with keys: matched, mismatched, missing, accuracy, details.
    """
    extracted_values = {}
    for e in extraction.enm_entries:
        extracted_values[e.entity_id] = e.value
    for t in extraction.triples:
        if t.numeric_value is not None:
            key = f"{t.head}_{t.relation}"
            extracted_values[key] = t.numeric_value

    matched = 0
    mismatched = 0
    missing = 0
    details = []

    for gt_key, gt_val in ground_truth.items():
        # Prefer exact key match; fall back to suffix/substring match
        # so ground-truth schemas that use short keys (e.g. "accuracy")
        # still resolve against fully-qualified extracted keys
        # (e.g. "rf_100_test_accuracy").
        ex_val = extracted_values.get(gt_key)
        if ex_val is None:
            # Suffix preferred over substring to avoid short-key collisions
            suffix_hits = [(k, v) for k, v in extracted_values.items()
                           if k.endswith("_" + gt_key) or k == gt_key]
            if suffix_hits:
                ex_val = suffix_hits[0][1]
            else:
                sub_hits = [(k, v) for k, v in extracted_values.items()
                            if gt_key in k]
                if sub_hits:
                    ex_val = sub_hits[0][1]

        if ex_val is None:
            missing += 1
            details.append((gt_key, gt_val, None, "MISSING"))
        elif abs(ex_val - gt_val) < 1e-4:
            matched += 1
            details.append((gt_key, gt_val, ex_val, "MATCH"))
        else:
            mismatched += 1
            details.append((gt_key, gt_val, ex_val, "MISMATCH"))

    total = matched + mismatched + missing
    accuracy = matched / total if total > 0 else 0.0

    return {
        "matched": matched,
        "mismatched": mismatched,
        "missing": missing,
        "accuracy": round(accuracy, 4),
        "total": total,
        "details": details,
    }
