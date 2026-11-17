# SPDX-License-Identifier: Apache-2.0
"""Sklearn tool definitions and execution for the governed agent.

Follows the same pattern as agents/ds_agent/tools.py:
  - TOOL_DEFINITIONS: list of JSON-schema dicts
  - execute_tool(name, args, state) -> (display_string, raw_dict)
  - Every raw_dict is pure Python (no numpy arrays, no DataFrames,
    no sklearn objects) so Tier 1 extraction works in Chapter 4.

Built in Chapter 3 of "How to Build Policy-Governed AI Agents."
"""

import traceback
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import (
    fetch_california_housing,
    fetch_openml,
    load_breast_cancer,
    load_iris,
    load_wine,
)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from knowlytix.harness.governance.session import SessionState

# ── Helpers ───────────────────────────────────────────────────────

def _to_python(obj) -> Any:
    """Recursively convert numpy/pandas types to pure Python."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 6)
    if isinstance(obj, np.ndarray):
        return [_to_python(x) for x in obj]
    if isinstance(obj, pd.DataFrame):
        return {col: [_to_python(v) for v in obj[col]] for col in obj.columns}
    if isinstance(obj, pd.Series):
        return {str(k): _to_python(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {str(k): _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(x) for x in obj]
    return obj


def _clf_metrics(y_true, y_pred, y_prob=None):
    """Classification metrics returned as a pure-Python dict."""
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
    }
    if y_prob is not None:
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] > 2:
                metrics["auc_ovr"] = round(float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr")
                ), 4)
            elif y_prob.ndim == 2 and y_prob.shape[1] == 2:
                metrics["auc"] = round(float(
                    roc_auc_score(y_true, y_prob[:, 1])
                ), 4)
            else:
                metrics["auc"] = round(float(
                    roc_auc_score(y_true, y_prob)
                ), 4)
        except Exception:
            pass
    return metrics


def _reg_metrics(y_true, y_pred):
    """Regression metrics returned as a pure-Python dict."""
    return {
        "mse": round(float(mean_squared_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


# ── Tool Definitions ──────────────────────────────────────────────

TOOL_DEFINITIONS = [
    # ── Data Tools ──
    {
        "name": "load_dataset",
        "description": (
            "Load a dataset via sklearn.datasets or CSV. "
            "Built-in: iris, wine, breast_cancer, california_housing, adult."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Dataset identifier for the session",
                },
                "builtin": {
                    "type": "string",
                    "enum": ["iris", "wine", "breast_cancer",
                             "california_housing", "adult"],
                    "description": "Built-in dataset name",
                },
                "csv_path": {
                    "type": "string",
                    "description": "Path to CSV file (alternative to builtin)",
                },
                "target_column": {
                    "type": "string",
                    "description": "Target column name (required for CSV)",
                },
                "test_size": {
                    "type": "number",
                    "description": "Test split ratio (default 0.2)",
                },
                "random_state": {
                    "type": "integer",
                    "description": "Random seed (default 42)",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "data_summary",
        "description": (
            "Summary statistics: df.describe(), dtypes and missing percentages."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset": {"type": "string", "description": "Dataset name"},
            },
            "required": ["dataset"],
        },
    },
    {
        "name": "eda_correlation",
        "description": "Compute correlation matrix between features.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset": {"type": "string"},
                "method": {
                    "type": "string",
                    "enum": ["pearson", "spearman"],
                    "description": "Correlation method (default: pearson)",
                },
            },
            "required": ["dataset"],
        },
    },
    {
        "name": "preprocess_data",
        "description": (
            "Impute + scale + encode via sklearn Pipeline. "
            "Stores fitted pipeline in state.preprocessors."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset": {"type": "string"},
                "impute_method": {
                    "type": "string",
                    "enum": ["mean", "median"],
                    "description": "Imputation strategy (default: mean)",
                },
                "scale_method": {
                    "type": "string",
                    "enum": ["standard", "minmax", "none"],
                    "description": "Scaling method (default: standard)",
                },
                "encode_method": {
                    "type": "string",
                    "enum": ["ordinal", "onehot", "none"],
                    "description": "Categorical encoding (default: ordinal)",
                },
            },
            "required": ["dataset"],
        },
    },
    # ── Model Tools ──
    {
        "name": "feature_select",
        "description": (
            "Feature selection via SelectKBest or tree-based importance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset": {"type": "string"},
                "method": {
                    "type": "string",
                    "enum": ["mutual_info", "f_classif", "tree"],
                    "description": "Selection method",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of features to select (default: 5)",
                },
            },
            "required": ["dataset", "method"],
        },
    },
    {
        "name": "train_model",
        "description": (
            "Fit an sklearn estimator and compute train/test metrics. "
            "Model types: LogisticRegression, RandomForestClassifier, "
            "GradientBoostingClassifier, DecisionTreeClassifier, SVC, "
            "RandomForestRegressor, GradientBoostingRegressor, "
            "LinearRegression, Ridge, Lasso."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset": {"type": "string"},
                "model_type": {"type": "string"},
                "model_name": {
                    "type": "string",
                    "description": "Unique name for the trained model",
                },
                "params": {
                    "type": "object",
                    "description": "Hyperparameters passed to the constructor",
                },
            },
            "required": ["dataset", "model_type", "model_name"],
        },
    },
    {
        "name": "predict",
        "description": "Generate predictions on the test set for a trained model.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset": {"type": "string"},
                "model_name": {"type": "string"},
            },
            "required": ["dataset", "model_name"],
        },
    },
    {
        "name": "leaderboard",
        "description": "Compare all trained models for a dataset, ranked by metric.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset": {"type": "string"},
                "sort_by": {
                    "type": "string",
                    "description": "Metric to sort by (default: test_accuracy or test_r2)",
                },
            },
            "required": ["dataset"],
        },
    },
    # ── Diagnostic Tools ──
    {
        "name": "diagnose_robustness",
        "description": (
            "Add Gaussian noise at multiple levels and re-score the model."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset": {"type": "string"},
                "model_name": {"type": "string"},
                "noise_levels": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Noise standard deviations (default: [0.01, 0.05, 0.1, 0.2])",
                },
            },
            "required": ["dataset", "model_name"],
        },
    },
    {
        "name": "diagnose_fairness",
        "description": (
            "Compute Adverse Impact Ratio (AIR) and demographic parity "
            "across groups defined by a protected attribute."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset": {"type": "string"},
                "model_name": {"type": "string"},
                "protected_attribute": {
                    "type": "string",
                    "description": "Feature name of the protected attribute",
                },
            },
            "required": ["dataset", "model_name", "protected_attribute"],
        },
    },
    {
        "name": "explain_importance",
        "description": "Permutation feature importance on test set.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset": {"type": "string"},
                "model_name": {"type": "string"},
                "n_repeats": {
                    "type": "integer",
                    "description": "Number of permutation repeats (default: 10)",
                },
            },
            "required": ["dataset", "model_name"],
        },
    },
    # ── Explanation Tools ──
    {
        "name": "explain_pdp",
        "description": "Partial dependence for selected features.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset": {"type": "string"},
                "model_name": {"type": "string"},
                "features": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Feature names for PDP (one or two)",
                },
            },
            "required": ["dataset", "model_name", "features"],
        },
    },
    {
        "name": "explain_shap",
        "description": "SHAP values: mean absolute SHAP per feature.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset": {"type": "string"},
                "model_name": {"type": "string"},
                "n_samples": {
                    "type": "integer",
                    "description": "Background samples for SHAP (default: 50)",
                },
            },
            "required": ["dataset", "model_name"],
        },
    },
    # ── Agent / Memory Tools (stubs) ──
    {
        "name": "recall_memory",
        "description": (
            "Query GMS store for relevant knowledge. "
            "Stub: full implementation in Chapter 5."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {"type": "string"},
                "relation": {"type": "string"},
            },
            "required": ["entity"],
        },
    },
    {
        "name": "save_finding",
        "description": (
            "Commit a finding (list of triples) to GMS. "
            "Stub: full implementation in Chapter 5."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "triples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {"type": "string"},
                            "object": {"type": "string"},
                        },
                    },
                    "description": "List of (subject, predicate, object) triples",
                },
            },
            "required": ["triples"],
        },
    },
]


# ── Tool Execution ────────────────────────────────────────────────

def execute_tool(name: str, arguments: dict, state: SessionState) -> tuple[str, dict]:
    """Execute a tool and return (display_string, raw_dict).

    The display string is a human-readable summary.
    The raw dict is pure Python and suitable for Tier 1 extraction.
    """
    try:
        return _dispatch(name, arguments, state)
    except Exception as e:
        tb = traceback.format_exc()
        return f"ERROR: {e}\n{tb}", {"error": str(e)}


def _dispatch(name: str, args: dict, state: SessionState):
    # Data tools
    if name == "load_dataset":
        return _load_dataset(args, state)
    elif name == "data_summary":
        return _data_summary(args, state)
    elif name == "eda_correlation":
        return _eda_correlation(args, state)
    elif name == "preprocess_data":
        return _preprocess_data(args, state)

    # Model tools
    elif name == "feature_select":
        return _feature_select(args, state)
    elif name == "train_model":
        return _train_model(args, state)
    elif name == "predict":
        return _predict(args, state)
    elif name == "leaderboard":
        return _leaderboard(args, state)

    # Diagnostic tools
    elif name == "diagnose_robustness":
        return _diagnose_robustness(args, state)
    elif name == "diagnose_fairness":
        return _diagnose_fairness(args, state)
    elif name == "explain_importance":
        return _explain_importance(args, state)

    # Explanation tools
    elif name == "explain_pdp":
        return _explain_pdp(args, state)
    elif name == "explain_shap":
        return _explain_shap(args, state)

    # Agent / memory stubs
    elif name == "recall_memory":
        return _recall_memory(args, state)
    elif name == "save_finding":
        return _save_finding(args, state)

    else:
        return f"Unknown tool: {name}", {"error": f"Unknown tool: {name}"}


# ── Data Implementations ──────────────────────────────────────────

_BUILTIN_LOADERS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "california_housing": fetch_california_housing,
}


def _load_dataset(args: dict, state: SessionState) -> tuple[str, dict]:
    """Load dataset and store in session."""
    name = args["name"]
    builtin = args.get("builtin", name)
    test_size = args.get("test_size", 0.2)
    random_state = args.get("random_state", 42)

    if builtin == "adult":
        # Fetch Adult Income from OpenML
        data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
        df = data.data.copy()
        target_name = "income"
        df[target_name] = (data.target == ">50K").astype(int)
        feature_names = [c for c in df.columns if c != target_name]
        task_type = "classification"
    elif builtin in _BUILTIN_LOADERS:
        loader = _BUILTIN_LOADERS[builtin]
        data = loader()
        feature_names = list(data.feature_names)
        target_name = "target"
        df = pd.DataFrame(data.data, columns=feature_names)
        df[target_name] = data.target
        task_type = ("regression" if builtin == "california_housing"
                     else "classification")
    elif "csv_path" in args:
        df = pd.read_csv(args["csv_path"])
        target_name = args.get("target_column", df.columns[-1])
        feature_names = [c for c in df.columns if c != target_name]
        nunique = df[target_name].nunique()
        task_type = "classification" if nunique <= 20 else "regression"
    else:
        return (f"ERROR: Unknown builtin '{builtin}'. "
                "Use iris, wine, breast_cancer, california_housing or adult.",
                {"error": f"Unknown builtin: {builtin}"})

    X = df[feature_names]
    y = df[target_name]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    state.datasets[name] = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "target_name": target_name,
        "task_type": task_type,
        "df": df,
    }

    n_samples = len(df)
    n_features = len(feature_names)
    display = (
        f"Loaded '{name}': {n_samples} samples, {n_features} features, "
        f"target='{target_name}', task={task_type}\n"
        f"  Train: {len(X_train)}, Test: {len(X_test)}"
    )
    raw = {
        "name": name,
        "n_samples": n_samples,
        "n_features": n_features,
        "target_name": target_name,
        "task_type": task_type,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_names": feature_names,
    }
    return display, raw


def _data_summary(args: dict, state: SessionState) -> tuple[str, dict]:
    """Per-feature statistics."""
    ds = state.datasets[args["dataset"]]
    df = ds["df"]
    desc = df.describe(include="all")
    dtypes = {col: str(dt) for col, dt in df.dtypes.items()}
    missing_pct = {col: round(float(df[col].isna().mean() * 100), 2)
                   for col in df.columns}

    display = (
        f"Data summary for '{args['dataset']}':\n"
        f"{desc.to_string()}\n\n"
        f"Missing %: {missing_pct}"
    )
    raw = {
        "describe": _to_python(desc.to_dict()),
        "dtypes": dtypes,
        "missing_pct": missing_pct,
        "n_rows": len(df),
        "n_cols": len(df.columns),
    }
    return display, raw


def _eda_correlation(args: dict, state: SessionState) -> tuple[str, dict]:
    """Correlation matrix."""
    ds = state.datasets[args["dataset"]]
    df = ds["df"]
    method = args.get("method", "pearson")

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(method=method)

    display = f"Correlation matrix ({method}):\n{corr.to_string()}"
    raw = {
        "method": method,
        "correlation": _to_python(corr.to_dict()),
    }
    return display, raw


def _preprocess_data(args: dict, state: SessionState) -> tuple[str, dict]:
    """Build and fit a preprocessing Pipeline."""
    ds = state.datasets[args["dataset"]]
    impute = args.get("impute_method", "mean")
    scale = args.get("scale_method", "standard")
    encode = args.get("encode_method", "ordinal")

    X_train = ds["X_train"].copy()
    X_test = ds["X_test"].copy()

    num_cols = list(X_train.select_dtypes(include=[np.number]).columns)
    cat_cols = list(X_train.select_dtypes(exclude=[np.number]).columns)

    n_features_before = X_train.shape[1]

    # Build numeric pipeline
    num_steps = [("imputer", SimpleImputer(strategy=impute))]
    if scale == "standard":
        num_steps.append(("scaler", StandardScaler()))
    elif scale == "minmax":
        num_steps.append(("scaler", MinMaxScaler()))
    num_pipe = Pipeline(num_steps)

    # Build categorical pipeline
    cat_steps = [("imputer", SimpleImputer(strategy="most_frequent"))]
    if encode == "ordinal":
        cat_steps.append(("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1)))
    elif encode == "onehot":
        cat_steps.append(("encoder", OneHotEncoder(
            handle_unknown="ignore", sparse_output=False)))
    cat_pipe = Pipeline(cat_steps) if cat_cols else None

    # Compose
    transformers = [("num", num_pipe, num_cols)]
    if cat_pipe and cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))

    preprocessor = ColumnTransformer(transformers, remainder="passthrough")
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    n_features_after = X_train_t.shape[1]

    # Store transformed data back
    ds["X_train_processed"] = X_train_t
    ds["X_test_processed"] = X_test_t
    state.preprocessors[args["dataset"]] = preprocessor

    display = (
        f"Preprocessing done: impute={impute}, scale={scale}, encode={encode}\n"
        f"  Features: {n_features_before} -> {n_features_after}\n"
        f"  Numeric cols: {len(num_cols)}, Categorical cols: {len(cat_cols)}"
    )
    raw = {
        "impute_method": impute,
        "scale_method": scale,
        "encode_method": encode,
        "n_features_before": n_features_before,
        "n_features_after": n_features_after,
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
    }
    return display, raw


# ── Model Implementations ─────────────────────────────────────────

def _get_model_class(model_type: str):
    """Lazy import of sklearn model classes."""
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    registry = {
        "LogisticRegression": LogisticRegression,
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "SVC": SVC,
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "DecisionTreeRegressor": DecisionTreeRegressor,
        "LinearRegression": LinearRegression,
        "Ridge": Ridge,
        "Lasso": Lasso,
    }
    if model_type not in registry:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {list(registry.keys())}"
        )
    return registry[model_type]


def _feature_select(args: dict, state: SessionState) -> tuple[str, dict]:
    """Feature selection via SelectKBest or tree-based importance."""
    ds = state.datasets[args["dataset"]]
    method = args["method"]
    k = args.get("k", 5)

    X_train = ds["X_train"]
    y_train = ds["y_train"]

    # Use only numeric columns for selection
    X_num = X_train.select_dtypes(include=[np.number])
    feature_names = list(X_num.columns)
    k = min(k, len(feature_names))

    if method == "f_classif":
        selector = SelectKBest(f_classif, k=k)
        selector.fit(X_num, y_train)
        scores = selector.scores_
    elif method == "mutual_info":
        selector = SelectKBest(mutual_info_classif, k=k)
        selector.fit(X_num, y_train)
        scores = selector.scores_
    elif method == "tree":
        from sklearn.ensemble import RandomForestClassifier as RFC
        rf = RFC(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_num, y_train)
        scores = rf.feature_importances_
    else:
        return f"Unknown method: {method}", {"error": f"Unknown method: {method}"}

    # Rank and select top-k
    ranked = sorted(zip(feature_names, scores), key=lambda x: -x[1])
    selected = [f for f, _ in ranked[:k]]
    importances = {f: round(float(s), 6) for f, s in ranked[:k]}

    display = (
        f"Feature selection ({method}, k={k}):\n"
        + "\n".join(f"  {f}: {importances[f]:.4f}" for f in selected)
    )
    raw = {
        "method": method,
        "k": k,
        "selected_features": selected,
        "importances": importances,
    }
    return display, raw


def _train_model(args: dict, state: SessionState) -> tuple[str, dict]:
    """Fit an sklearn estimator and compute metrics."""
    ds = state.datasets[args["dataset"]]
    model_type = args["model_type"]
    model_name = args["model_name"]
    params = args.get("params", {})

    cls = _get_model_class(model_type)
    estimator = cls(**params)

    X_train = ds["X_train"]
    X_test = ds["X_test"]
    y_train = ds["y_train"]
    y_test = ds["y_test"]

    # Use numeric-only columns
    num_cols = list(X_train.select_dtypes(include=[np.number]).columns)
    X_tr = X_train[num_cols].values
    X_te = X_test[num_cols].values

    estimator.fit(X_tr, y_train)

    y_pred_train = estimator.predict(X_tr)
    y_pred_test = estimator.predict(X_te)

    task = ds["task_type"]
    if task == "classification":
        y_prob_train = None
        y_prob_test = None
        if hasattr(estimator, "predict_proba"):
            y_prob_train = estimator.predict_proba(X_tr)
            y_prob_test = estimator.predict_proba(X_te)

        train_metrics = _clf_metrics(y_train, y_pred_train, y_prob_train)
        test_metrics = _clf_metrics(y_test, y_pred_test, y_prob_test)
        train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
        test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
    else:
        train_m = _reg_metrics(y_train, y_pred_train)
        test_m = _reg_metrics(y_test, y_pred_test)
        train_metrics = {f"train_{k}": v for k, v in train_m.items()}
        test_metrics = {f"test_{k}": v for k, v in test_m.items()}

    all_metrics = {**train_metrics, **test_metrics}

    # Compute train-test gap
    if task == "classification":
        gap = round(train_metrics.get("train_accuracy", 0)
                    - test_metrics.get("test_accuracy", 0), 4)
    else:
        gap = round(train_metrics.get("train_r2", 0)
                    - test_metrics.get("test_r2", 0), 4)
    all_metrics["train_test_gap"] = gap

    # Store
    state.models[model_name] = {
        "estimator": estimator,
        "model_type": model_type,
        "params": _to_python(params),
        "dataset": args["dataset"],
        "num_cols": num_cols,
    }
    state.results[model_name] = all_metrics

    display = (
        f"Trained '{model_name}' ({model_type}):\n"
        + "\n".join(f"  {k}: {v}" for k, v in all_metrics.items())
    )
    raw = {
        "model_name": model_name,
        "model_type": model_type,
        "metrics": all_metrics,
    }
    return display, raw


def _predict(args: dict, state: SessionState) -> tuple[str, dict]:
    """Generate predictions for a trained model."""
    ds = state.datasets[args["dataset"]]
    model_info = state.models[args["model_name"]]
    estimator = model_info["estimator"]
    num_cols = model_info["num_cols"]

    X_test = ds["X_test"][num_cols].values
    y_pred = estimator.predict(X_test)

    pred_dict = {
        "y_pred": _to_python(y_pred),
        "n_predictions": len(y_pred),
    }

    if hasattr(estimator, "predict_proba"):
        y_prob = estimator.predict_proba(X_test)
        pred_dict["y_prob_shape"] = list(y_prob.shape)

    state.predictions[args["model_name"]] = pred_dict

    display = (
        f"Predictions for '{args['model_name']}': "
        f"{len(y_pred)} samples, "
        f"unique values: {len(set(_to_python(y_pred)))}"
    )
    raw = {
        "model_name": args["model_name"],
        "n_predictions": len(y_pred),
        "unique_values": sorted(list(set(_to_python(y_pred)))),
    }
    return display, raw


def _leaderboard(args: dict, state: SessionState) -> tuple[str, dict]:
    """Rank all trained models for a given dataset."""
    dataset_name = args["dataset"]
    sort_by = args.get("sort_by")

    rows = []
    for mname, minfo in state.models.items():
        if minfo["dataset"] != dataset_name:
            continue
        metrics = state.results.get(mname, {})
        rows.append({"model": mname, "type": minfo["model_type"], **metrics})

    if not rows:
        return "No trained models for this dataset.", {"models": []}

    # Determine default sort column
    if sort_by is None:
        task = state.datasets[dataset_name]["task_type"]
        sort_by = "test_accuracy" if task == "classification" else "test_r2"

    rows.sort(key=lambda r: r.get(sort_by, 0), reverse=True)

    header = f"{'Rank':<5} {'Model':<25} {'Type':<30} {sort_by:<15}"
    lines = [header, "-" * len(header)]
    for i, r in enumerate(rows, 1):
        val = r.get(sort_by, "N/A")
        lines.append(f"{i:<5} {r['model']:<25} {r['type']:<30} {val}")

    display = "Leaderboard:\n" + "\n".join(lines)
    raw = {
        "sort_by": sort_by,
        "ranking": [
            {"rank": i + 1, "model": r["model"], "type": r["type"],
             sort_by: r.get(sort_by), "metrics": {k: v for k, v in r.items()
                                                    if k not in ("model", "type")}}
            for i, r in enumerate(rows)
        ],
    }
    return display, raw


# ── Diagnostic Implementations ────────────────────────────────────

def _diagnose_robustness(args: dict, state: SessionState) -> tuple[str, dict]:
    """Perturb each input with Gaussian noise scaled to its training std, re-score.

    Noise at level sigma means sigma * per-feature-std, so the same sigma
    grid is interpretable across datasets with heterogeneous feature scales.
    The raw dict carries (i) per-level scores, (ii) a baseline-vs-worst drop,
    (iii) a verdict (ROBUST/SENSITIVE/FRAGILE) and (iv) explicit coverage
    bookkeeping so downstream policy consumers can tell which features were
    actually probed and which the model never saw.
    """
    ds = state.datasets[args["dataset"]]
    model_info = state.models[args["model_name"]]
    estimator = model_info["estimator"]
    num_cols = model_info["num_cols"]
    noise_levels = args.get("noise_levels", [0.01, 0.05, 0.1, 0.2])

    X_train = ds["X_train"][num_cols].values
    X_test = ds["X_test"][num_cols].values
    y_test = ds["y_test"]
    task = ds["task_type"]

    # Per-feature std (floor at 1e-9 to avoid divide-by-zero on constants)
    feature_stds = np.maximum(X_train.std(axis=0), 1e-9)

    metric_name = "accuracy" if task == "classification" else "r2"
    metric_fn = accuracy_score if task == "classification" else r2_score

    baseline = round(float(metric_fn(y_test, estimator.predict(X_test))), 4)

    rng = np.random.RandomState(42)
    results = {}
    for sigma in noise_levels:
        noise = rng.normal(0, sigma, X_test.shape) * feature_stds
        score = round(float(metric_fn(y_test, estimator.predict(X_test + noise))), 4)
        results[f"noise_{sigma}"] = {metric_name: score}

    scored = [v[metric_name] for v in results.values()]
    max_drop = round(baseline - min(scored), 4)

    if max_drop <= 0.01:
        verdict = "ROBUST"
    elif max_drop <= 0.05:
        verdict = "SENSITIVE"
    else:
        verdict = "FRAGILE"

    all_features = list(ds["X_test"].columns)
    skipped = [c for c in all_features if c not in num_cols]

    key = f"{args['model_name']}_robustness"
    state.results[key] = results

    lines = [f"Robustness diagnosis for '{args['model_name']}':"]
    lines.append("  Scale:     noise sigma * per-feature-std")
    lines.append(f"  Coverage:  {len(num_cols)}/{len(all_features)} features "
                 f"(model uses only numeric)")
    if skipped:
        preview = ", ".join(skipped[:4])
        tail = f" (+{len(skipped)-4} more)" if len(skipped) > 4 else ""
        lines.append(f"  Untested:  {preview}{tail}")
    lines.append(f"  Baseline {metric_name}: {baseline}")
    for k, v in results.items():
        lines.append(f"  {k}: {v}")
    lines.append(f"  Max drop:  {max_drop}")
    lines.append(f"  Verdict:   {verdict}")
    display = "\n".join(lines)

    raw = {
        "model_name": args["model_name"],
        "baseline": baseline,
        "noise_results": results,
        "max_drop": max_drop,
        "verdict": verdict,
        "coverage": {
            "scale_mode": "per_feature_std",
            "n_features_noised": len(num_cols),
            "n_features_total": len(all_features),
            "features_noised": list(num_cols),
            "features_skipped_by_model": skipped,
        },
    }
    return display, raw


def _diagnose_fairness(args: dict, state: SessionState) -> tuple[str, dict]:
    """Compute AIR and demographic parity across protected groups."""
    ds = state.datasets[args["dataset"]]
    model_info = state.models[args["model_name"]]
    estimator = model_info["estimator"]
    num_cols = model_info["num_cols"]
    protected = args["protected_attribute"]

    X_test = ds["X_test"]
    y_test = ds["y_test"]

    if protected not in X_test.columns:
        # Check if it is in the full df
        df = ds["df"]
        if protected not in df.columns:
            return (f"ERROR: '{protected}' not found in dataset.",
                    {"error": f"'{protected}' not found"})
        # Use the protected attribute values aligned with test indices
        prot_values = df.loc[X_test.index, protected]
    else:
        prot_values = X_test[protected]

    X_te = X_test[num_cols].values
    y_pred = estimator.predict(X_te)

    groups = prot_values.unique()
    group_rates = {}
    group_accuracy = {}

    for g in groups:
        mask = (prot_values == g).values
        if mask.sum() == 0:
            continue
        g_pred = y_pred[mask]
        g_true = y_test.values[mask] if hasattr(y_test, 'values') else y_test[mask]
        # Positive prediction rate (for classification)
        pos_rate = float(g_pred.mean()) if len(g_pred) > 0 else 0.0
        acc = float(accuracy_score(g_true, g_pred))
        group_rates[str(g)] = round(pos_rate, 4)
        group_accuracy[str(g)] = round(acc, 4)

    # Adverse Impact Ratio = min_rate / max_rate
    rates = list(group_rates.values())
    if max(rates) > 0:
        air = round(min(rates) / max(rates), 4)
    else:
        air = 0.0

    # Demographic parity difference
    dp_diff = round(max(rates) - min(rates), 4)

    fairness = {
        "protected_attribute": protected,
        "group_positive_rates": group_rates,
        "group_accuracy": group_accuracy,
        "adverse_impact_ratio": air,
        "demographic_parity_diff": dp_diff,
        "air_passes_4_5_rule": air >= 0.8,
    }

    key = f"{args['model_name']}_fairness"
    state.results[key] = fairness

    lines = [f"Fairness diagnosis for '{args['model_name']}' on '{protected}':"]
    lines.append(f"  Group positive rates: {group_rates}")
    lines.append(f"  Group accuracy:       {group_accuracy}")
    lines.append(f"  Adverse Impact Ratio: {air}")
    lines.append(f"  Demographic parity diff: {dp_diff}")
    lines.append(f"  AIR >= 0.8 (4/5 rule): {air >= 0.8}")
    display = "\n".join(lines)

    return display, fairness


def _explain_importance(args: dict, state: SessionState) -> tuple[str, dict]:
    """Permutation feature importance."""
    ds = state.datasets[args["dataset"]]
    model_info = state.models[args["model_name"]]
    estimator = model_info["estimator"]
    num_cols = model_info["num_cols"]
    n_repeats = args.get("n_repeats", 10)

    X_test = ds["X_test"][num_cols].values
    y_test = ds["y_test"]

    result = permutation_importance(
        estimator, X_test, y_test,
        n_repeats=n_repeats, random_state=42, n_jobs=-1,
    )

    importance = {
        col: round(float(result.importances_mean[i]), 6)
        for i, col in enumerate(num_cols)
    }
    # Sort descending
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

    key = f"{args['model_name']}_importance"
    state.results[key] = importance

    lines = [f"Permutation importance for '{args['model_name']}':"]
    for feat, imp in importance.items():
        lines.append(f"  {feat}: {imp:.4f}")
    display = "\n".join(lines)

    raw = {
        "model_name": args["model_name"],
        "n_repeats": n_repeats,
        "importances": importance,
    }
    return display, raw


# ── Explanation Implementations ───────────────────────────────────

def _explain_pdp(args: dict, state: SessionState) -> tuple[str, dict]:
    """Partial dependence."""
    ds = state.datasets[args["dataset"]]
    model_info = state.models[args["model_name"]]
    estimator = model_info["estimator"]
    num_cols = model_info["num_cols"]
    features = args["features"]

    # Cast to float64 — partial_dependence builds a grid of interpolated
    # values and rejects integer-dtype columns (raises TypeError in sklearn
    # when it tries to assign a float grid point into an int column).
    X_train = ds["X_train"][num_cols].astype(float)

    # Map feature names to column indices
    feat_indices = [num_cols.index(f) for f in features if f in num_cols]
    if not feat_indices:
        return ("ERROR: Features not found in numeric columns.",
                {"error": "Features not found"})

    result = partial_dependence(
        estimator, X_train, features=feat_indices,
        kind="average", grid_resolution=20,
    )

    pdp_values = {}
    for i, fname in enumerate(features):
        if i < len(result["grid_values"]):
            grid = _to_python(result["grid_values"][i])
            avg = _to_python(result["average"][0] if len(features) == 1
                             else result["average"][i])
            pdp_values[fname] = {
                "grid": grid[:10],  # first 10 for display
                "average": avg[:10] if isinstance(avg, list) else avg,
            }

    display = (
        f"Partial dependence for '{args['model_name']}' on {features}:\n"
        + "\n".join(f"  {f}: grid[0..9]={pdp_values[f]['grid']}"
                    for f in pdp_values)
    )
    raw = {
        "model_name": args["model_name"],
        "features": features,
        "pdp": pdp_values,
    }
    return display, raw


def _explain_shap(args: dict, state: SessionState) -> tuple[str, dict]:
    """SHAP values: mean absolute SHAP per feature."""
    ds = state.datasets[args["dataset"]]
    model_info = state.models[args["model_name"]]
    estimator = model_info["estimator"]
    num_cols = model_info["num_cols"]
    n_samples = args.get("n_samples", 50)

    X_test = ds["X_test"][num_cols].values
    # Subsample for speed
    n = min(n_samples, len(X_test))
    X_bg = X_test[:n]

    try:
        import shap
        # Use KernelExplainer for model-agnostic SHAP
        if hasattr(estimator, "predict_proba"):
            explainer = shap.KernelExplainer(estimator.predict_proba, X_bg)
        else:
            explainer = shap.KernelExplainer(estimator.predict, X_bg)
        shap_values = explainer.shap_values(X_bg)

        # Handle multi-class: take mean across classes
        shap_arr = np.abs(np.array(shap_values))
        if shap_arr.ndim == 3:
            # (n_samples, n_features, n_classes) -> average over classes
            shap_arr = np.mean(shap_arr, axis=2)

        mean_shap = np.mean(shap_arr, axis=0)
        shap_dict = {
            col: round(float(mean_shap[i]), 6)
            for i, col in enumerate(num_cols)
        }
        shap_dict = dict(sorted(shap_dict.items(), key=lambda x: -x[1]))

    except ImportError:
        return ("ERROR: shap package not installed.",
                {"error": "shap not installed"})

    display = (
        f"Mean |SHAP| for '{args['model_name']}' (n={n}):\n"
        + "\n".join(f"  {f}: {v:.4f}" for f, v in shap_dict.items())
    )
    raw = {
        "model_name": args["model_name"],
        "n_samples": n,
        "mean_abs_shap": shap_dict,
    }
    return display, raw


# ── Agent / Memory Stubs ──────────────────────────────────────────

def _recall_memory(args: dict, state: SessionState) -> tuple[str, dict]:
    """Stub for GMS memory recall (Chapter 5)."""
    entity = args.get("entity", "")
    relation = args.get("relation", "")
    display = (
        f"[STUB] recall_memory(entity='{entity}', relation='{relation}')\n"
        "Full implementation in Chapter 5."
    )
    raw = {
        "entity": entity,
        "relation": relation,
        "triples": [],
        "stub": True,
    }
    return display, raw


def _save_finding(args: dict, state: SessionState) -> tuple[str, dict]:
    """Stub for GMS finding storage (Chapter 5)."""
    triples = args.get("triples", [])
    display = (
        f"[STUB] save_finding: {len(triples)} triples received.\n"
        "Full implementation in Chapter 5."
    )
    raw = {
        "n_triples": len(triples),
        "triples": _to_python(triples),
        "stub": True,
    }
    return display, raw
