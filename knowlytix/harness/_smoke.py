# SPDX-License-Identifier: Apache-2.0
"""GMS-096: post-install acceptance test, surfaced as the ``knowlytix-smoke``
console script.

Mirrors the external PyPI flow: after ``pip install knowlytix-harness``, a user runs
``knowlytix-smoke`` to verify their install and ``knowlytix-smoke --notebook-path``
to locate the bundled exploration notebook.

Two callers share this module:

* ``knowlytix-smoke`` — the installed console script declared in
  ``packaging/knowlytix-harness/pyproject.toml`` ``[project.scripts]``.  Canonical
  entry point; matches what external PyPI users hit.
* ``scripts/smoke_test.py`` — thin editable-checkout wrapper that
  imports :func:`main` directly.  Lets in-repo CI
  (``packaging/verify_wheels.sh``, pre-push hooks) run against a
  source tree without a ``pip install -e``.

``--notebook-path`` resolves ``harness/smoke/smoke.ipynb``, which
ships as package data inside the ``knowlytix-harness`` wheel (GMS-097).

Design goals:

* **Key-free** — no provider API keys, no GPU, no network. Anything that
  needs LiteLLM is stubbed or skipped. This module must pass on a
  minimal CI runner.
* **Fast** — every step completes in well under a second on commodity
  hardware. Heavy end-to-end flows belong in the Jupyter notebook
  shipped alongside at ``harness/smoke/smoke.ipynb``.
* **Actionable failures** — each assertion names the package and the
  specific invariant that broke, so the first line of the traceback
  tells you what to fix.

Exit codes:

* ``0`` — all checks passed.
* ``1`` — at least one assertion failed; traceback names the broken
  check.

See ``distribution/implementation_plan.md`` §GMS-057 (original smoke
script) and §GMS-096 (move into ``knowlytix-harness`` wheel as a console script).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from importlib.resources import files
from pathlib import Path

# Tuple (not frozenset): insertion order drives the banner output so
# a regression in step-1 ordering stays debuggable across Python builds.
__all__ = ["main", "notebook_path"]

PACKAGES: tuple[str, ...] = ("knowlytix.core", "knowlytix.knowledge", "knowlytix.benchmark", "knowlytix.harness.testing")

# Module-level contract tables — lifting these out of helper bodies
# keeps each `_check_*` helper under 20 lines and makes the shipped
# schemas greppable ("what does knowlytix-smoke require from a wheel?").
_FIXTURE_ASSETS: tuple[tuple[str, str], ...] = (
    ("knowlytix.knowledge.fixtures.smoke", "sample.md"),
    ("knowlytix.benchmark.fixtures.smoke", "questions.json"),
    ("knowlytix.benchmark.fixtures.smoke", "predictions.json"),
    ("knowlytix.harness.fixtures.smoke", "doe_smoke.json"),
    ("knowlytix.harness.fixtures.smoke", "ground_truth.json"),
)
_DOE_REQUIRED_KEYS: tuple[str, ...] = ("campaign", "factors", "scenarios")
_GROUND_TRUTH_REQUIRED_KEYS: tuple[str, ...] = ("entities", "relations", "facts")

_TOTAL_STEPS = 5


def notebook_path() -> Path:
    """Absolute, canonical path to the bundled ``smoke.ipynb``.

    Single source of truth shared by ``main()`` (prints the path for
    ``jupyter lab $(knowlytix-smoke --notebook-path)``) and
    ``tests/harness/test_smoke_notebook.py`` (asserts equality against
    what the CLI prints).  Keeps the Traversable-to-Path coercion +
    symlink resolution in one place if importlib.resources semantics
    tighten in a future Python version.
    """
    return Path(str(files("knowlytix.harness.smoke") / "smoke.ipynb")).resolve()


def _step(n: int, msg: str) -> None:
    """Print a stepped progress line.  Output shape is grep-friendly and
    matches the banner shape teammates have learned since GMS-057.
    """
    print(f"[{n}/{_TOTAL_STEPS}] {msg}")


def _require(cond: bool, msg: str) -> None:
    """Assertion helper that survives ``python -O``.

    The shipped ``knowlytix-smoke`` entry point is user-facing; a user who
    runs their venv with ``-O`` (or PYTHONOPTIMIZE=1) would silently
    skip every bare ``assert`` and see the success banner regardless of
    actual state.  Raising explicitly keeps the contract honest.
    """
    if not cond:
        raise AssertionError(msg)


def _check_imports_and_all() -> None:
    """Step 1: every package imports and its ``__all__`` resolves."""
    for name in PACKAGES:
        mod = importlib.import_module(name)
        _require(hasattr(mod, "__all__"), f"{name}: missing __all__")
        _require(bool(mod.__all__), f"{name}: __all__ is empty")
        for sym in mod.__all__:
            _require(
                hasattr(mod, sym),
                f"{name}.{sym} declared in __all__ but not resolvable",
            )
    _step(1, f"imports + __all__ resolved for {len(PACKAGES)} packages")


def _settings_classes() -> tuple[type, ...]:
    """Return the five Settings classes spanning the four wheels.

    Kept in a function (not a module-level constant) because the imports
    themselves are part of what step 2 is testing — eager imports at
    module load would crash before the smoke banner even starts.
    """
    from knowlytix.core import GMSCoreSettings, LLMSettings
    from knowlytix.knowledge import DocGMSSettings
    from knowlytix.benchmark import FinStructBenchSettings
    from knowlytix.harness.settings import GMSHSettings

    return (
        GMSCoreSettings,
        LLMSettings,
        DocGMSSettings,
        FinStructBenchSettings,
        GMSHSettings,
    )


def _check_settings_instantiation() -> None:
    """Step 2: Settings classes instantiate with their defaults.

    Covers GMS-015..018 env-var plumbing: if the pydantic-settings
    schema drifted the wheel would crash here instead of at runtime.
    """
    classes = _settings_classes()
    for cls in classes:
        instance = cls()
        _require(instance is not None, f"{cls.__name__}() returned None")
    _step(2, f"{len(classes)} Settings classes instantiate with defaults")


def _check_fixture_resources() -> None:
    """Step 3: smoke fixtures reachable via ``importlib.resources``.

    Mirrors the README quickstarts — if MANIFEST shipping drifted the
    files would be missing from the wheel and resolve as empty bytes.
    """
    for pkg, name in _FIXTURE_ASSETS:
        path = files(pkg) / name
        data = path.read_bytes()
        _require(
            len(data) > 0,
            f"{pkg}/{name} is empty in the installed wheel — MANIFEST may be wrong",
        )
    _step(3, f"{len(_FIXTURE_ASSETS)} fixtures reachable via importlib.resources")


# JSON scalars loaded from `predictions.json` / `questions.json` —
# narrower than `object` so float()/int() calls in `_coerce_answer`
# don't need type:ignore escape hatches.
_JsonScalar = str | int | float | bool | None


def _coerce_answer(raw: _JsonScalar, gt: object) -> object:
    """Coerce ``raw`` to ground-truth's type so the scorer exercises the
    float-tolerance / int / bool parse paths rather than defaulting to
    string-compare.

    Bool check MUST precede int — Python's ``isinstance(True, int)``
    returns True, so a bool ground truth would match the int branch
    and get parsed via ``int()`` instead of the truthy-string path.
    """
    if isinstance(gt, bool):
        return str(raw).strip().lower() in ("true", "1", "yes")
    if isinstance(gt, float):
        try:
            return float(raw) if raw is not None else raw
        except (TypeError, ValueError):
            return raw
    if isinstance(gt, int):
        try:
            return int(raw) if raw is not None else raw
        except (TypeError, ValueError):
            return raw
    return raw


def _load_score_fixtures() -> tuple[list[dict[str, object]], dict[str, _JsonScalar]]:
    """Return the shipped questions list + predictions-by-id map.

    Narrows the json-decoded ``object`` graph back to concrete shapes
    via runtime ``isinstance`` checks so downstream code (and
    ``_coerce_answer``'s ``_JsonScalar``-typed ``raw`` parameter) sees
    proper types with no ``type: ignore`` escape hatches.
    """
    q_doc = json.loads(
        (files("knowlytix.benchmark.fixtures.smoke") / "questions.json").read_text()
    )
    p_doc = json.loads(
        (files("knowlytix.benchmark.fixtures.smoke") / "predictions.json").read_text()
    )
    _require(isinstance(q_doc, dict), "questions.json top-level must be a JSON object")
    _require(isinstance(p_doc, dict), "predictions.json top-level must be a JSON object")
    questions = q_doc["questions"]
    predictions = p_doc["predictions"]
    _require(isinstance(questions, list), "questions.json 'questions' must be a list")
    _require(isinstance(predictions, list), "predictions.json 'predictions' must be a list")

    preds_by_id: dict[str, _JsonScalar] = {}
    for p in predictions:
        _require(isinstance(p, dict), "prediction entries must be JSON objects")
        pid = p["id"]
        _require(isinstance(pid, str), "prediction 'id' must be a string")
        preds_by_id[pid] = p["answer"]
    return questions, preds_by_id


def _check_score_answer() -> None:
    """Step 4: ``knowlytix.benchmark.score_answer`` — the key-free entry point."""
    from knowlytix.benchmark import ScoreResult, score_answer

    questions, preds_by_id = _load_score_fixtures()
    scored = 0
    for q in questions:
        qid = q["id"]
        _require(isinstance(qid, str), f"question 'id' must be str, got {type(qid).__name__}")
        raw = preds_by_id.get(qid)
        if raw is None:
            continue
        answer = _coerce_answer(raw, q["ground_truth"])
        result = score_answer(answer, q["ground_truth"])
        _require(
            isinstance(result, ScoreResult),
            f"score_answer returned {type(result).__name__}, expected ScoreResult",
        )
        scored += 1
    _require(
        scored == len(questions),
        f"knowlytix.benchmark: scored {scored} of {len(questions)} smoke questions",
    )
    _step(4, f"knowlytix.benchmark.score_answer() ran on {scored} smoke questions")


def _check_doe_fixtures() -> None:
    """Step 5: harness DOE fixtures parse and match the expected schema.

    Running a full DOE campaign needs an LLM; here we just assert the
    fixtures look like what the ``knowlytix-harness`` CLI expects so the quickstart
    in ``packaging/knowlytix-harness/README.md`` keeps working after a wheel rebuild.
    """
    doe_smoke: dict[str, object] = json.loads(
        (files("knowlytix.harness.fixtures.smoke") / "doe_smoke.json").read_text()
    )
    ground_truth: dict[str, object] = json.loads(
        (files("knowlytix.harness.fixtures.smoke") / "ground_truth.json").read_text()
    )
    for key in _DOE_REQUIRED_KEYS:
        _require(
            key in doe_smoke,
            f"harness doe_smoke.json missing '{key}' — schema drifted vs GMS-037",
        )
    for key in _GROUND_TRUTH_REQUIRED_KEYS:
        _require(
            key in ground_truth,
            f"harness ground_truth.json missing '{key}' — schema drifted vs GMS-037",
        )
    _step(5, "harness DOE smoke fixtures parse and match expected schema")


def _run_assertions() -> int:
    """Execute the 5-step smoke suite; return exit code.

    Each step is a private helper that owns its own `_step(...)` banner
    so the orchestrator stays a readable checklist.
    """
    _check_imports_and_all()
    _check_settings_instantiation()
    _check_fixture_resources()
    _check_score_answer()
    _check_doe_fixtures()

    print()
    print("=" * 60)
    print("Knowlytix GMS wheels — post-install smoke test PASSED")
    print("=" * 60)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``knowlytix-smoke`` console script.

    Default mode runs the 5-step assertion suite.  ``--notebook-path``
    prints the absolute path to ``harness/smoke/smoke.ipynb`` (the
    companion notebook shipped as ``knowlytix-harness`` package data).
    """
    parser = argparse.ArgumentParser(
        prog="knowlytix-smoke",
        description=(
            "Post-install smoke test for the four Knowlytix GMS wheels. "
            "Runs key-free assertions in <1s; exit 0 on success."
        ),
    )
    parser.add_argument(
        "--notebook-path",
        action="store_true",
        help=(
            "Print the absolute path to the bundled exploration "
            "notebook and exit 0 without running the smoke assertions. "
            "Pipe into jupyter: `jupyter lab $(knowlytix-smoke --notebook-path)`."
        ),
    )
    args = parser.parse_args(argv)

    if args.notebook_path:
        print(notebook_path())
        return 0

    return _run_assertions()


if __name__ == "__main__":
    sys.exit(main())
