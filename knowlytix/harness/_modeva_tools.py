# SPDX-License-Identifier: Apache-2.0
"""MoDeva tool definitions and execution for the Claude tool_use API.

Public home (GMS-171) — relocated from the private `agents.ds_agent.tools`
tree. `modeva` itself remains an optional dep (not listed in
`packaging/knowlytix-harness/pyproject.toml`): when it isn't installed,
`SessionState` + `TOOL_DEFINITIONS` + `execute_tool` still import
cleanly, but tool invocation raises a clear error pointing the user
at Knowlytix commercial support. This matches the current public-wheel
semantics — MoDeva tooling is Knowlytix-internal; public harness users
register their own tools via `ToolRegistry`.

Internal callers keep working via the re-export shim at
`agents/ds_agent/tools.py` in the private monorepo.
"""

import json
import traceback
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    from modeva import DataSet, ModelZoo, TestSuite  # type: ignore[import-not-found]
    _MODEVA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without modeva installed
    DataSet = ModelZoo = TestSuite = None  # type: ignore[assignment,misc]
    _MODEVA_AVAILABLE = False


# ── Session State ──────────────────────────────────────────────

@dataclass
class SessionState:
    """Mutable state persisting across tool calls within a session."""
    datasets: dict = field(default_factory=dict)        # name -> DataSet
    model_zoos: dict = field(default_factory=dict)      # name -> ModelZoo
    trained_models: dict = field(default_factory=dict)   # name -> model object
    results: dict = field(default_factory=dict)          # key -> ValidationResult


# ── Model Registry ─────────────────────────────────────────────

def _build_model_registry():
    """Lazily build model class registry."""
    from modeva.models import (
        MoXGBRegressor, MoXGBClassifier,
        MoLGBMRegressor, MoLGBMClassifier,
        MoCatBoostRegressor, MoCatBoostClassifier,
        MoRandomForestRegressor, MoRandomForestClassifier,
        MoGradientBoostingRegressor, MoGradientBoostingClassifier,
        MoDecisionTreeRegressor, MoDecisionTreeClassifier,
        MoElasticNet, MoLogisticRegression,
        MoGAMINetRegressor, MoGAMINetClassifier,
        MoReLUDNNRegressor, MoReLUDNNClassifier,
    )
    return {
        "MoXGBRegressor": MoXGBRegressor,
        "MoXGBClassifier": MoXGBClassifier,
        "MoLGBMRegressor": MoLGBMRegressor,
        "MoLGBMClassifier": MoLGBMClassifier,
        "MoCatBoostRegressor": MoCatBoostRegressor,
        "MoCatBoostClassifier": MoCatBoostClassifier,
        "MoRandomForestRegressor": MoRandomForestRegressor,
        "MoRandomForestClassifier": MoRandomForestClassifier,
        "MoGradientBoostingRegressor": MoGradientBoostingRegressor,
        "MoGradientBoostingClassifier": MoGradientBoostingClassifier,
        "MoDecisionTreeRegressor": MoDecisionTreeRegressor,
        "MoDecisionTreeClassifier": MoDecisionTreeClassifier,
        "MoElasticNet": MoElasticNet,
        "MoLogisticRegression": MoLogisticRegression,
        "MoGAMINetRegressor": MoGAMINetRegressor,
        "MoGAMINetClassifier": MoGAMINetClassifier,
        "MoReLUDNNRegressor": MoReLUDNNRegressor,
        "MoReLUDNNClassifier": MoReLUDNNClassifier,
    }


MODEL_REGISTRY = None


def _get_registry():
    global MODEL_REGISTRY
    if MODEL_REGISTRY is None:
        MODEL_REGISTRY = _build_model_registry()
    return MODEL_REGISTRY


# ── JSON Encoding ──────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        return super().default(obj)


def _serialize_result(result) -> str:
    """Serialize a ValidationResult to a string for Claude."""
    parts = []
    if result is None:
        return "No result."

    if hasattr(result, "table") and result.table is not None:
        tbl = result.table
        if isinstance(tbl, pd.DataFrame):
            parts.append(f"TABLE:\n{tbl.to_markdown(index=True)}")
        elif isinstance(tbl, dict):
            for k, v in tbl.items():
                if isinstance(v, pd.DataFrame):
                    md = v.to_markdown(index=True)
                    parts.append(f"TABLE ({k}):\n{md}")
                else:
                    parts.append(f"TABLE ({k}): {v}")

    if hasattr(result, "value") and result.value is not None:
        try:
            val = result.value
            # Only serialize non-huge values
            val_str = json.dumps(val, cls=NumpyEncoder, indent=2, default=str)
            if len(val_str) < 5000:
                parts.append(f"VALUES:\n{val_str}")
            else:
                # Summarize large values
                if isinstance(val, dict):
                    keys = list(val.keys())
                    parts.append(f"VALUE KEYS: {keys}")
        except Exception:
            parts.append("VALUES: [serialization error]")

    return "\n\n".join(parts) if parts else "Done (no tabular output)."


# ── Tool Definitions ───────────────────────────────────────────

TOOL_DEFINITIONS = [
    # ── Data ──
    {
        "name": "load_dataset",
        "description": (
            "Load a dataset, set the target variable, and create train/test split. "
            "Built-in datasets: BikeSharing, TaiwanCredit, SimuCredit, CaliforniaHousing. "
            "Or provide csv_path for custom data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Dataset name (used as identifier)",
                },
                "builtin": {
                    "type": "string",
                    "enum": ["BikeSharing", "TaiwanCredit", "SimuCredit",
                             "CaliforniaHousing"],
                    "description": "Built-in dataset name. Omit if using csv_path.",
                },
                "csv_path": {
                    "type": "string",
                    "description": "Path to CSV file. Omit if using builtin.",
                },
                "target": {
                    "type": "string",
                    "description": "Target column name. Required for CSV data.",
                },
                "test_ratio": {
                    "type": "number",
                    "description": "Test split ratio (default 0.2)",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "data_summary",
        "description": "Get summary statistics of the loaded dataset including feature types, missing values, and distributions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
            },
            "required": ["dataset_name"],
        },
    },
    {
        "name": "eda_correlation",
        "description": "Compute correlation matrix between features. Methods: pearson, spearman, kendall, xicor.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "method": {
                    "type": "string",
                    "enum": ["pearson", "spearman", "kendall", "xicor"],
                    "description": "Correlation method (default: pearson)",
                },
            },
            "required": ["dataset_name"],
        },
    },
    {
        "name": "preprocess_data",
        "description": (
            "Preprocess data: impute missing values, scale numerical features, "
            "encode categorical features. Call this before training models."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "impute_method": {
                    "type": "string",
                    "enum": ["mean", "median", "most_frequent"],
                    "description": "Imputation method (default: mean)",
                },
                "scale_method": {
                    "type": "string",
                    "enum": ["minmax", "standardize", "quantile", "none"],
                    "description": "Scaling method (default: minmax)",
                },
                "encode_method": {
                    "type": "string",
                    "enum": ["ordinal", "onehot", "target", "none"],
                    "description": "Categorical encoding (default: ordinal)",
                },
            },
            "required": ["dataset_name"],
        },
    },
    {
        "name": "feature_engineer",
        "description": (
            "Apply feature engineering transformers. Available: bivector, wedge_bivector, "
            "whitening, shear, phase_encoder, density, rf_proximity, direct_rs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "methods": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of FE methods to apply",
                },
            },
            "required": ["dataset_name", "methods"],
        },
    },
    {
        "name": "feature_select",
        "description": "Run feature selection. Methods: correlation, xgbpfi, rcit.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "method": {
                    "type": "string",
                    "enum": ["correlation", "xgbpfi", "rcit"],
                },
                "threshold": {"type": "number"},
            },
            "required": ["dataset_name", "method"],
        },
    },
    # ── Model ──
    {
        "name": "add_model",
        "description": (
            "Add a model to the model zoo. Available model types: "
            "MoXGBRegressor, MoXGBClassifier, MoLGBMRegressor, MoLGBMClassifier, "
            "MoCatBoostRegressor, MoCatBoostClassifier, MoRandomForestRegressor, "
            "MoRandomForestClassifier, MoDecisionTreeRegressor, MoDecisionTreeClassifier, "
            "MoElasticNet, MoLogisticRegression, MoGAMINetRegressor, MoGAMINetClassifier, "
            "MoReLUDNNRegressor, MoReLUDNNClassifier. "
            "For FANOVA interpretation, use tree models with max_depth=2."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "model_type": {"type": "string", "description": "Model class name"},
                "model_name": {"type": "string", "description": "Unique model name"},
                "params": {
                    "type": "object",
                    "description": "Model hyperparameters (e.g. max_depth, n_estimators)",
                },
            },
            "required": ["dataset_name", "model_type", "model_name"],
        },
    },
    {
        "name": "train_model",
        "description": "Train a single model that has been added to the model zoo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "model_name": {"type": "string"},
            },
            "required": ["dataset_name", "model_name"],
        },
    },
    {
        "name": "train_all_models",
        "description": "Train all models in the model zoo for a dataset.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
            },
            "required": ["dataset_name"],
        },
    },
    {
        "name": "leaderboard",
        "description": "Show model comparison leaderboard with all metrics for trained models.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
            },
            "required": ["dataset_name"],
        },
    },
    # ── Diagnose ──
    {
        "name": "diagnose_accuracy",
        "description": (
            "Compute accuracy metrics on train and test sets. "
            "Regression: MSE, MAE, R2. Classification: AUC, F1, ACC, LogLoss. "
            "Shows train-test gap to detect overfitting."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "model_name": {"type": "string"},
            },
            "required": ["dataset_name", "model_name"],
        },
    },
    {
        "name": "diagnose_robustness",
        "description": (
            "Test model stability under input perturbations at various noise levels. "
            "Identifies non-robust samples."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "model_name": {"type": "string"},
                "noise_levels": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Noise levels to test (default: [0.1])",
                },
            },
            "required": ["dataset_name", "model_name"],
        },
    },
    {
        "name": "diagnose_reliability",
        "description": (
            "Assess prediction reliability via conformal prediction. "
            "Reports coverage and interval width."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "model_name": {"type": "string"},
                "alpha": {
                    "type": "number",
                    "description": "Miscoverage rate (default: 0.1 for 90% coverage)",
                },
            },
            "required": ["dataset_name", "model_name"],
        },
    },
    {
        "name": "diagnose_resilience",
        "description": (
            "Test performance on progressively harder subsets. "
            "Methods: worst-sample, outer-sample, hard-sample, worst-cluster."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "model_name": {"type": "string"},
                "method": {
                    "type": "string",
                    "enum": ["worst-sample", "outer-sample", "hard-sample",
                             "worst-cluster"],
                },
            },
            "required": ["dataset_name", "model_name"],
        },
    },
    {
        "name": "diagnose_fairness",
        "description": (
            "Test fairness across demographic groups. Requires protected data set on dataset. "
            "Classification: AIR, PR, RR. Regression: SMD."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "model_name": {"type": "string"},
                "group_feature": {
                    "type": "string",
                    "description": "Protected feature name (e.g. 'Gender')",
                },
                "group_value": {
                    "type": "string",
                    "description": "Protected group value (e.g. 'Female' or '1')",
                },
            },
            "required": ["dataset_name", "model_name", "group_feature"],
        },
    },
    {
        "name": "diagnose_weakness",
        "description": (
            "Detect weakness regions using AMIF. Identifies 2D regions where the model "
            "performs worst, plus features that explain the weakness."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "model_name": {"type": "string"},
                "geometry_method": {
                    "type": "string",
                    "enum": ["arf", "carf"],
                    "description": "Geometry method (default: arf)",
                },
                "bins": {"type": "integer", "description": "Grid bins (default: 10)"},
            },
            "required": ["dataset_name", "model_name"],
        },
    },
    # ── Explain ──
    {
        "name": "explain_pfi",
        "description": "Permutation feature importance: measures how much error increases when each feature is shuffled.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "model_name": {"type": "string"},
            },
            "required": ["dataset_name", "model_name"],
        },
    },
    {
        "name": "explain_pdp",
        "description": "Partial dependence plot showing marginal effect of one or two features.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "model_name": {"type": "string"},
                "features": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One feature (1D) or two features (2D interaction)",
                },
            },
            "required": ["dataset_name", "model_name", "features"],
        },
    },
    {
        "name": "explain_shap",
        "description": "SHAP local explanation for a single sample.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "model_name": {"type": "string"},
                "sample_index": {"type": "integer", "description": "Sample index (default: 0)"},
            },
            "required": ["dataset_name", "model_name"],
        },
    },
    {
        "name": "explain_ale",
        "description": "Accumulated local effects: feature effects robust to correlated features.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "model_name": {"type": "string"},
                "features": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One or two features",
                },
            },
            "required": ["dataset_name", "model_name", "features"],
        },
    },
    # ── Compare ──
    {
        "name": "compare_accuracy",
        "description": "Compare accuracy metrics across multiple trained models.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "model_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of trained model names to compare",
                },
            },
            "required": ["dataset_name", "model_names"],
        },
    },
    {
        "name": "compare_robustness",
        "description": "Compare robustness across multiple trained models.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "model_names": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["dataset_name", "model_names"],
        },
    },
    # ── Interpret (FANOVA) ──
    {
        "name": "interpret_fi",
        "description": (
            "FANOVA feature importance from inherently interpretable models. "
            "Requires a model with FANOVA support (e.g., depth-2 tree ensemble, GAMINet)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "model_name": {"type": "string"},
            },
            "required": ["dataset_name", "model_name"],
        },
    },
    {
        "name": "interpret_effects",
        "description": (
            "Plot FANOVA main effect curves and interaction heatmaps. "
            "Shows exact shape of each feature's contribution."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "model_name": {"type": "string"},
                "features": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Features to plot (one for main effect, two for interaction). Omit for all.",
                },
            },
            "required": ["dataset_name", "model_name"],
        },
    },
    # ── Memory ──
    {
        "name": "recall_memory",
        "description": (
            "Query the geometric memory system for relevant knowledge. "
            "Searches both the MoDeva knowledge base (API/methodology) and "
            "session memory (prior experiment findings)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Question to ask the memory system"},
            },
            "required": ["query"],
        },
    },
]


# ── Tool Execution ─────────────────────────────────────────────

def execute_tool(name: str, arguments: dict, state: SessionState):
    """Execute a tool and return (result_string, raw_ValidationResult_or_None).

    The result_string goes back to Claude as tool_result content.
    The raw result is used for triple extraction into GMS.
    """
    if not _MODEVA_AVAILABLE:
        return (
            f"ERROR: MoDeva is not installed. Tool '{name}' is a MoDeva-backed "
            "tool and requires Knowlytix-internal tooling. Public harness users "
            "should register their own tools via `ToolRegistry.register(...)`.",
            None,
        )
    try:
        return _dispatch(name, arguments, state)
    except Exception as e:
        tb = traceback.format_exc()
        return f"ERROR: {e}\n{tb}", None


def _dispatch(name: str, args: dict, state: SessionState):
    # ── Data tools ──
    if name == "load_dataset":
        return _load_dataset(args, state)
    elif name == "data_summary":
        return _data_summary(args, state)
    elif name == "eda_correlation":
        return _eda_correlation(args, state)
    elif name == "preprocess_data":
        return _preprocess_data(args, state)
    elif name == "feature_engineer":
        return _feature_engineer(args, state)
    elif name == "feature_select":
        return _feature_select(args, state)

    # ── Model tools ──
    elif name == "add_model":
        return _add_model(args, state)
    elif name == "train_model":
        return _train_model(args, state)
    elif name == "train_all_models":
        return _train_all(args, state)
    elif name == "leaderboard":
        return _leaderboard(args, state)

    # ── Diagnose tools ──
    elif name == "diagnose_accuracy":
        return _diagnose_accuracy(args, state)
    elif name == "diagnose_robustness":
        return _diagnose_robustness(args, state)
    elif name == "diagnose_reliability":
        return _diagnose_reliability(args, state)
    elif name == "diagnose_resilience":
        return _diagnose_resilience(args, state)
    elif name == "diagnose_fairness":
        return _diagnose_fairness(args, state)
    elif name == "diagnose_weakness":
        return _diagnose_weakness(args, state)

    # ── Explain tools ──
    elif name == "explain_pfi":
        return _explain_pfi(args, state)
    elif name == "explain_pdp":
        return _explain_pdp(args, state)
    elif name == "explain_shap":
        return _explain_shap(args, state)
    elif name == "explain_ale":
        return _explain_ale(args, state)

    # ── Compare tools ──
    elif name == "compare_accuracy":
        return _compare_accuracy(args, state)
    elif name == "compare_robustness":
        return _compare_robustness(args, state)

    # ── Interpret tools ──
    elif name == "interpret_fi":
        return _interpret_fi(args, state)
    elif name == "interpret_effects":
        return _interpret_effects(args, state)

    # ── Memory (handled by agent.py) ──
    elif name == "recall_memory":
        return "Memory recall is handled by the agent layer.", None

    else:
        return f"Unknown tool: {name}", None


# ── Data Implementations ───────────────────────────────────────

def _load_dataset(args, state):
    ds_name = args["name"]
    ds = DataSet(name=ds_name)

    if "builtin" in args and args["builtin"]:
        ds.load(name=args["builtin"])
    elif "csv_path" in args and args["csv_path"]:
        ds.load_csv(args["csv_path"])
    else:
        return "ERROR: Provide either 'builtin' or 'csv_path'.", None

    if "target" in args and args["target"]:
        ds.set_target(args["target"])

    test_ratio = args.get("test_ratio", 0.2)
    ds.set_random_split(test_ratio=test_ratio)

    state.datasets[ds_name] = ds

    # Create ModelZoo
    mz = ModelZoo(dataset=ds)
    state.model_zoos[ds_name] = mz

    info = (
        f"Dataset '{ds_name}' loaded.\n"
        f"  Shape: {ds.shape}\n"
        f"  Task: {ds.task_type}\n"
        f"  Target: {ds.target_feature_name}\n"
        f"  Features: {ds.n_features} ({len(ds.feature_names_numerical)} numerical, "
        f"{len(ds.feature_names_categorical)} categorical)\n"
        f"  Train/Test split: {len(ds.train_idx)}/{len(ds.test_idx)}"
    )
    return info, None


def _data_summary(args, state):
    ds = state.datasets[args["dataset_name"]]
    result = ds.summary()
    state.results[f"{args['dataset_name']}_summary"] = result
    return _serialize_result(result), result


def _eda_correlation(args, state):
    ds = state.datasets[args["dataset_name"]]
    method = args.get("method", "pearson")
    result = ds.eda_correlation(method=method)
    return _serialize_result(result), result


def _preprocess_data(args, state):
    ds = state.datasets[args["dataset_name"]]
    ds.reset_preprocess()

    impute = args.get("impute_method", "mean")
    ds.impute_missing(method=impute)

    scale = args.get("scale_method", "minmax")
    if scale != "none":
        ds.scale_numerical(method=scale)

    encode = args.get("encode_method", "ordinal")
    if encode != "none" and ds.feature_names_categorical:
        ds.encode_categorical(method=encode)

    ds.preprocess()
    return f"Preprocessing done: impute={impute}, scale={scale}, encode={encode}.", None


def _feature_engineer(args, state):
    ds = state.datasets[args["dataset_name"]]
    ds.reset_feature_engineering()

    methods = args.get("methods", [])
    for m in methods:
        if m == "bivector":
            ds.fe_bivector(max_features=100, selection="variance")
        elif m == "wedge_bivector":
            ds.fe_wedge_bivector(max_features=100, selection="variance")
        elif m == "whitening":
            ds.fe_whitening()
        elif m == "shear":
            ds.fe_shear()
        elif m == "phase_encoder":
            ds.fe_phase_encoder(n_thresholds=5)
        elif m == "density":
            ds.fe_density()
        elif m == "rf_proximity":
            ds.fe_rf_proximity()
        elif m == "direct_rs":
            ds.fe_direct_rs()

    ds.engineer_features()
    return f"Feature engineering done: {methods}. New feature count: {ds.n_features}.", None


def _feature_select(args, state):
    ds = state.datasets[args["dataset_name"]]
    method = args["method"]
    threshold = args.get("threshold")

    if method == "correlation":
        result = ds.feature_select_corr(threshold=threshold or 0.2)
    elif method == "xgbpfi":
        result = ds.feature_select_xgbpfi(threshold=threshold or 0.1)
    elif method == "rcit":
        result = ds.feature_select_rcit(threshold=threshold or 1e-6)
    else:
        return f"Unknown feature selection method: {method}", None

    return _serialize_result(result), result


# ── Model Implementations ──────────────────────────────────────

def _add_model(args, state):
    ds_name = args["dataset_name"]
    model_type = args["model_type"]
    model_name = args["model_name"]
    params = args.get("params", {})

    registry = _get_registry()
    if model_type not in registry:
        return f"Unknown model type: {model_type}. Available: {list(registry.keys())}", None

    model_cls = registry[model_type]
    model = model_cls(name=model_name, **params)

    mz = state.model_zoos[ds_name]
    mz.add_model(model, name=model_name)

    return f"Model '{model_name}' ({model_type}) added with params={params}.", None


def _train_model(args, state):
    ds_name = args["dataset_name"]
    model_name = args["model_name"]
    mz = state.model_zoos[ds_name]
    mz.train(model_name)
    model = mz.get_model(model_name)
    state.trained_models[model_name] = model
    return f"Model '{model_name}' trained.", None


def _train_all(args, state):
    ds_name = args["dataset_name"]
    mz = state.model_zoos[ds_name]
    mz.train_all()

    for name in mz.list_model_names():
        state.trained_models[name] = mz.get_model(name)

    return f"All models trained: {mz.list_model_names()}", None


def _leaderboard(args, state):
    ds_name = args["dataset_name"]
    mz = state.model_zoos[ds_name]
    lb = mz.leaderboard()
    return f"LEADERBOARD:\n{lb.to_markdown(index=True)}", None


# ── Diagnose Implementations ──────────────────────────────────

def _get_ts(state, ds_name, model_name):
    ds = state.datasets[ds_name]
    model = state.trained_models[model_name]
    return TestSuite(dataset=ds, model=model)


def _get_ts_multi(state, ds_name, model_names):
    ds = state.datasets[ds_name]
    models = [state.trained_models[n] for n in model_names]
    return TestSuite(dataset=ds, models=models)


def _diagnose_accuracy(args, state):
    ts = _get_ts(state, args["dataset_name"], args["model_name"])
    result = ts.diagnose_accuracy_table()
    key = f"{args['model_name']}_accuracy"
    state.results[key] = result
    return _serialize_result(result), result


def _diagnose_robustness(args, state):
    ts = _get_ts(state, args["dataset_name"], args["model_name"])
    noise = args.get("noise_levels", [0.1])
    if isinstance(noise, list) and len(noise) == 1:
        noise = noise[0]
    result = ts.diagnose_robustness(noise_levels=noise)
    state.results[f"{args['model_name']}_robustness"] = result
    return _serialize_result(result), result


def _diagnose_reliability(args, state):
    ts = _get_ts(state, args["dataset_name"], args["model_name"])
    alpha = args.get("alpha", 0.1)
    result = ts.diagnose_reliability(alpha=alpha)
    state.results[f"{args['model_name']}_reliability"] = result
    return _serialize_result(result), result


def _diagnose_resilience(args, state):
    ts = _get_ts(state, args["dataset_name"], args["model_name"])
    method = args.get("method", "worst-sample")
    result = ts.diagnose_resilience(method=method)
    state.results[f"{args['model_name']}_resilience"] = result
    return _serialize_result(result), result


def _diagnose_fairness(args, state):
    ts = _get_ts(state, args["dataset_name"], args["model_name"])
    feature = args["group_feature"]
    value = args.get("group_value")
    if value is not None:
        try:
            value = [int(value)]
        except ValueError:
            value = [value]
    group_config = {feature: value} if value else {feature: [1]}
    result = ts.diagnose_fairness(group_config=group_config)
    state.results[f"{args['model_name']}_fairness"] = result
    return _serialize_result(result), result


def _diagnose_weakness(args, state):
    ts = _get_ts(state, args["dataset_name"], args["model_name"])
    geom = args.get("geometry_method", "arf")
    bins = args.get("bins", 10)
    result = ts.diagnose_weakness_region(geometry_method=geom, bins=bins)
    state.results[f"{args['model_name']}_weakness"] = result
    return _serialize_result(result), result


# ── Explain Implementations ────────────────────────────────────

def _explain_pfi(args, state):
    ts = _get_ts(state, args["dataset_name"], args["model_name"])
    result = ts.explain_pfi()
    state.results[f"{args['model_name']}_pfi"] = result
    return _serialize_result(result), result


def _explain_pdp(args, state):
    ts = _get_ts(state, args["dataset_name"], args["model_name"])
    features = args["features"]
    feat = features[0] if len(features) == 1 else tuple(features[:2])
    result = ts.explain_pdp(features=feat)
    return _serialize_result(result), result


def _explain_shap(args, state):
    ts = _get_ts(state, args["dataset_name"], args["model_name"])
    idx = args.get("sample_index", 0)
    result = ts.explain_shap(sample_index=idx)
    return _serialize_result(result), result


def _explain_ale(args, state):
    ts = _get_ts(state, args["dataset_name"], args["model_name"])
    features = args["features"]
    feat = features[0] if len(features) == 1 else tuple(features[:2])
    result = ts.explain_ale(features=feat)
    return _serialize_result(result), result


# ── Compare Implementations ────────────────────────────────────

def _compare_accuracy(args, state):
    ts = _get_ts_multi(state, args["dataset_name"], args["model_names"])
    result = ts.compare_accuracy_table()
    state.results["compare_accuracy"] = result
    return _serialize_result(result), result


def _compare_robustness(args, state):
    ts = _get_ts_multi(state, args["dataset_name"], args["model_names"])
    result = ts.compare_robustness()
    state.results["compare_robustness"] = result
    return _serialize_result(result), result


# ── Interpret Implementations ──────────────────────────────────

def _interpret_fi(args, state):
    ts = _get_ts(state, args["dataset_name"], args["model_name"])
    result = ts.interpret_fi()
    state.results[f"{args['model_name']}_interpret_fi"] = result
    return _serialize_result(result), result


def _interpret_effects(args, state):
    ts = _get_ts(state, args["dataset_name"], args["model_name"])
    features = args.get("features")
    if features:
        feat = features[0] if len(features) == 1 else tuple(features[:2])
    else:
        feat = None
    result = ts.interpret_effects(features=feat)
    return _serialize_result(result), result
