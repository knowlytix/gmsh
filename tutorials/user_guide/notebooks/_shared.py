"""Shared imports, paths, and plot helpers for User Guide tutorial notebooks."""

import warnings
warnings.filterwarnings("ignore")

import contextlib
import glob
import io
import json
import math
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TUTORIALS_ROOT = REPO_ROOT / "tutorials" / "user_guide"
FIXTURES_DIR = TUTORIALS_ROOT / "fixtures"
NOTEBOOKS_DIR = TUTORIALS_ROOT / "notebooks"
FIGURES_DIR = NOTEBOOKS_DIR / "_figures"


def bootstrap_runtime(repo: str = "~/jupyterlab/kg-memory") -> None:
    """Prepend cluster-venv site-packages and repo to sys.path; stub modeva if absent."""
    for p in glob.glob("/home/asudjianto/cluster/spark-venv/lib/python*/site-packages"):
        if p not in sys.path:
            sys.path.insert(0, p)
    repo_abs = os.path.expanduser(repo)
    if repo_abs not in sys.path:
        sys.path.insert(0, repo_abs)
    import importlib
    try:
        importlib.import_module("modeva")
    except ImportError:
        import types
        stub = types.ModuleType("modeva")

        class _Placeholder:
            def __init__(self, *_, **__):
                raise RuntimeError("modeva not installed; stub forbids instantiation.")

        stub.DataSet = _Placeholder
        stub.ModelZoo = _Placeholder
        stub.TestSuite = _Placeholder
        sys.modules["modeva"] = stub


bootstrap_runtime()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextlib.contextmanager
def quiet():
    """Suppress stdout during noisy GMS training and ingestion calls."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def _load_store(store_dir: Path, device: torch.device | None = None):
    """Internal: rehydrate a GMSExpertStore from a fixture directory."""
    if not store_dir.exists():
        raise FileNotFoundError(
            f"Fixture not found: {store_dir}. Run "
            "`python -m user_guide_tutorials.scripts.build_fixture` first."
        )
    from knowlytix.knowledge.config import DocGMSConfig
    from knowlytix.knowledge.store import GMSExpertStore

    dev = device or DEVICE
    config = DocGMSConfig(store_path=str(store_dir))
    store = GMSExpertStore(config, device=dev)
    with quiet():
        ok = store.load()
    if not ok:
        raise RuntimeError(f"Store at {store_dir} could not be loaded.")
    return store


def load_ds_practices_store(device: torch.device | None = None):
    """Load the prebuilt data-science-practices fixture store."""
    return _load_store(FIXTURES_DIR / "ds_practices_store", device)


def load_california_housing_store(device: torch.device | None = None):
    """Load the prebuilt California-housing-domain fixture store."""
    return _load_store(FIXTURES_DIR / "california_housing_store", device)


def load_adult_income_store(device: torch.device | None = None):
    """Alias kept for plan compatibility; resolves to california_housing_store."""
    return load_california_housing_store(device)


def plot_triple_score_hist(scores_true, scores_false, ax=None, title=None,
                           threshold=1.0):
    """Overlapping histograms of triple scores with a shaded geodesic threshold."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0.0, max(2.0, max(scores_true + scores_false, default=2.0)), 30)
    ax.hist(scores_true, bins=bins, alpha=0.6, label="known true", color="tab:blue")
    ax.hist(scores_false, bins=bins, alpha=0.6, label="fabricated", color="tab:red")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1)
    ax.axvspan(0.0, threshold, alpha=0.05, color="tab:blue")
    ax.axvspan(threshold, ax.get_xlim()[1], alpha=0.05, color="tab:red")
    ax.set_xlabel("geodesic score (lower is more plausible)")
    ax.set_ylabel("count")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right")
    return ax


def plot_tension_spectrum(values, ax=None, title=None):
    """Single-axis histogram of tension energy with three banded shadings."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    sqrt2 = math.sqrt(2.0)
    bins = np.linspace(0.0, 2.0, 40)
    ax.hist(values, bins=bins, color="tab:gray", edgecolor="black", alpha=0.8)
    ax.axvspan(0.0, 0.5, alpha=0.10, color="tab:green", label="agreement")
    ax.axvspan(0.5, sqrt2, alpha=0.10, color="tab:olive", label="irrelevance")
    ax.axvspan(sqrt2, 2.0, alpha=0.10, color="tab:red", label="contradiction")
    ax.set_xlabel("tension energy")
    ax.set_ylabel("count")
    ax.set_xlim(0.0, 2.0)
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_signal_radar(signal_vector, ax=None, label=None, title=None):
    """Six-component radar of (sim, ver, depth, nov, margin, signal_unavailable)."""
    import matplotlib.pyplot as plt

    keys = ["sim", "ver", "depth", "nov", "margin", "signal_unavailable"]
    if hasattr(signal_vector, "__dict__"):
        vals = [float(getattr(signal_vector, k, 0.0) or 0.0) for k in keys]
    elif isinstance(signal_vector, dict):
        vals = [float(signal_vector.get(k, 0.0) or 0.0) for k in keys]
    else:
        vals = [float(v or 0.0) for v in list(signal_vector)[:6]]
        vals = vals + [0.0] * (6 - len(vals))
    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    vals_loop = vals + vals[:1]
    angles_loop = angles + angles[:1]
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})
    ax.plot(angles_loop, vals_loop, color="tab:blue", linewidth=1.5, label=label)
    ax.fill(angles_loop, vals_loop, color="tab:blue", alpha=0.15)
    ax.set_xticks(angles)
    ax.set_xticklabels(keys, fontsize=8)
    ax.set_ylim(0.0, 1.0)
    if title:
        ax.set_title(title)
    if label:
        ax.legend(loc="upper right", fontsize=8)
    return ax


def assert_block_executed(nb_path: Path) -> bool:
    """Return True iff every code cell in the notebook has at least one output."""
    nb_path = Path(nb_path)
    with open(nb_path) as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if not "".join(cell.get("source", [])).strip():
            continue
        outputs = cell.get("outputs", [])
        if not outputs:
            return False
    return True


def manifest() -> dict:
    """Read the fixtures manifest emitted by build_fixture.py."""
    path = FIXTURES_DIR / "MANIFEST.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)
