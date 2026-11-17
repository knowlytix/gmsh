# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for GMSH testing user-guide tutorial notebooks.

Every notebook in `knowlytix/harness/testing/user_guide_tutorials/notebooks/`
should start with:

    from _shared import load_ch04_store, setup_plotting, seed_all
    store = load_ch04_store()
    setup_plotting()
    seed_all()

All notebooks read the hardened ground-truth store at
`gmsh_testing_ext/lab_stores/ch04_ground_truth/` read-only. Do not mutate.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any

# File path: knowlytix/harness/testing/user_guide_tutorials/notebooks/_shared.py
# Climb 5 levels (notebooks → user_guide_tutorials → testing → harness → knowlytix → repo).
REPO_ROOT = Path(__file__).resolve().parents[5]
# Backwards-compat alias: upstream uses KG_ROOT (from its dev-box layout at
# ~/jupyterlab/kg-memory). Fork-side this is just the repo root. Re-exported
# for notebook + build-script cells that still spell it KG_ROOT.
KG_ROOT = REPO_ROOT
LAB_STORES_PATH = Path(__file__).resolve().parents[1] / "lab_stores"
CH04_STORE_PATH = LAB_STORES_PATH / "ch04_ground_truth"
CALIBRATION_BASE_PATH = REPO_ROOT / "gmsh_testing_ext" / "lab_stores" / "calibration_base"
CALIBRATION_DOCQ1_PATH = REPO_ROOT / "gmsh_testing_ext" / "lab_stores" / "ch18_doc_q1"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def assert_fixture_available() -> None:
    """Fail fast with a clear message if the ch04 fixture is missing."""
    required = [
        CH04_STORE_PATH / "adapter.json",
        CH04_STORE_PATH / "model.pt",
        CH04_STORE_PATH / "triples.json",
        CH04_STORE_PATH / "enm.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Ground-truth fixture missing files:\n  "
            + "\n  ".join(missing)
            + f"\n\nExpected location: {CH04_STORE_PATH}\n"
            "This fixture ships with the monograph at gmsh_testing_ext/. "
            "If you are running outside the monograph repo, see "
            "knowlytix/harness/testing/user_guide_tutorials/README.md."
        )


def load_ch04_store():
    """Load the canonical ch04 ground-truth store. Read-only.

    Returns:
        GMSExpertStore with 514 entities, 68 relations, 823 triples,
        cap_enabled=True. Do not mutate.
    """
    assert_fixture_available()
    from knowlytix.knowledge.config import DocGMSConfig
    from knowlytix.knowledge.store import GMSExpertStore

    config = DocGMSConfig(store_path=str(CH04_STORE_PATH))
    store = GMSExpertStore(config)
    store.load()
    return store


def load_calibration(name: str = "base") -> dict[str, Any]:
    """Load a hardened calibration artifact.

    Args:
        name: "base" for `calibration_base/calibration.json`,
              "doc_q1" for `ch18_doc_q1/calibration.json`.

    Returns:
        Parsed JSON dict with τ_geo, τ_ent, τ_contra, etc.
    """
    lookup = {
        "base": CALIBRATION_BASE_PATH / "calibration.json",
        "doc_q1": CALIBRATION_DOCQ1_PATH / "calibration.json",
    }
    if name not in lookup:
        raise ValueError(f"Unknown calibration name: {name!r}. Choices: {list(lookup)}")
    path = lookup[name]
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration artifact missing: {path}\n"
            "These ship with the monograph. Check gmsh_testing_ext/lab_stores/."
        )
    return json.loads(path.read_text())


def setup_plotting() -> None:
    """Configure matplotlib defaults for tutorial figures."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 100,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
    })


def seed_all(seed: int = 42) -> None:
    """Pin every RNG we care about for reproducible tutorial output."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


DEFAULT_TUTORIAL_MODEL = "claude-sonnet-4-6"


def require_anthropic_key() -> str:
    """Resolve the Anthropic API key via the project's canonical resolver.

    Resolution order (defined in knowlytix.harness.governance.api_key):
      1. ANTHROPIC_API_KEY environment variable
      2. `pass show anthropic/api-key` (Unix password store)

    Raises AnthropicAPIKeyError on miss. No mock-LLM fallback.
    """
    from knowlytix.harness.governance.api_key import get_anthropic_api_key
    key = get_anthropic_api_key()
    # Mirror into os.environ so harness internals that read ANTHROPIC_API_KEY
    # directly (e.g. anthropic.Anthropic() default ctor) pick it up.
    os.environ["ANTHROPIC_API_KEY"] = key
    return key


def make_anthropic_client():
    """Return a configured Anthropic client. Raises on missing key."""
    require_anthropic_key()
    import anthropic
    return anthropic.Anthropic()


def store_summary(store) -> dict[str, Any]:
    """Compact one-line summary for tutorial intros."""
    return store.stats()
