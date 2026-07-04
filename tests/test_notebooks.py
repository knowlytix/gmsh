"""
Executes notebooks in:
  - knowlytix/harness/governance/tutorials/notebooks/
  - knowlytix/harness/testing/tutorials/notebooks/

Fails if any cell produces an error output. All cells run even when one raises
(allow_errors), so a single test reports every failure in the notebook — not
just the first.

Run all notebooks:
    pytest tests/

Run a single notebook:
    pytest tests/ -k governance/01_prerequisites
    pytest tests/ -k testing/01_framing

Run all notebooks in one suite:
    pytest tests/ -k "governance/"
    pytest tests/ -k "testing/"

Run with verbose cell-error output:
    pytest tests/ -v
"""

import pytest
import nbformat
from nbclient import NotebookClient
from pathlib import Path

_here = Path(__file__).resolve().parent
# Dual-layout root detection:
#   GMS source repository: tests/gmsh/test_notebooks.py → _here.name == "gmsh"
#   gmsh mirror:  tests/test_notebooks.py      → _here.name == "tests"
_root = _here.parent if _here.name == "tests" else _here.parent.parent

_NOTEBOOK_DIRS = [
    _root / "knowlytix" / "harness" / "governance" / "tutorials" / "notebooks",
    _root / "knowlytix" / "harness" / "testing" / "tutorials" / "notebooks",
    # keyless-demo quickstart. It sets KNOWLYTIX_DEMO_MODE=1 in its
    # first cell, so it runs identically here (licensed CI kernel) and in a
    # zero-credential gmsh clone — proving the keyless path in the same matrix.
    _root / "knowlytix" / "harness" / "demo",
]


def _suite_id(nb: Path) -> str:
    """Stable test id: ``<suite>/<stem>`` regardless of nesting depth.

    Tutorial notebooks live at ``.../<suite>/tutorials/notebooks/X.ipynb``;
    the demo notebook lives directly under ``.../demo/X.ipynb``.
    """
    parent = nb.parent.name
    suite = nb.parent.parent.parent.name if parent == "notebooks" else parent
    return f"{suite}/{nb.stem}"

KERNEL = "gms-ci"
TIMEOUT = 900  # seconds per cell

notebooks = sorted(
    nb
    for d in _NOTEBOOK_DIRS
    if d.is_dir()
    for nb in d.glob("*.ipynb")
)


@pytest.mark.parametrize(
    "notebook",
    notebooks,
    ids=[_suite_id(nb) for nb in notebooks],
)
def test_notebook(notebook):
    nb = nbformat.read(notebook, as_version=4)

    client = NotebookClient(
        nb,
        timeout=TIMEOUT,
        kernel_name=KERNEL,
        allow_errors=True,
        resources={"metadata": {"path": str(notebook.parent)}},
    )
    client.execute()

    errors = []
    for idx, cell in enumerate(nb.cells, 1):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                etype = output.get("ename", "Error")
                emsg = output.get("evalue", "")
                errors.append(f"cell {idx}: {etype}: {emsg}")

    assert not errors, "\n" + "\n".join(errors)
