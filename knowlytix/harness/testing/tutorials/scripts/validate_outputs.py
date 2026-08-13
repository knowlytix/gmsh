# SPDX-License-Identifier: Apache-2.0
"""Per-notebook validation for user_guide_tutorials/notebooks/.

For each NB, checks:
  1. Every code cell that is expected to produce a visible output has outputs
     (defensive against silently-broken cells after a partial re-execution).
  2. NB-specific spot-checks: the canonical headline numbers each NB is
     supposed to surface remain in the executed outputs.

Usage:
    cd ~/jupyterlab/kg-memory
    ~/cluster/spark-venv/bin/python knowlytix/harness/testing/user_guide_tutorials/scripts/validate_outputs.py
    ~/cluster/spark-venv/bin/python knowlytix/harness/testing/user_guide_tutorials/scripts/validate_outputs.py --only 00
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
NOTEBOOKS_DIR = (
    REPO_ROOT / "harness" / "doe" / "user_guide_tutorials" / "notebooks"
)
NB_RE = re.compile(r"^(\d{2})_.*\.ipynb$")

# ---------------------------------------------------------------------------
# Generic checks
# ---------------------------------------------------------------------------

_OUTPUT_CUES = (
    "print(", "display(", "plt.show", "plt.subplot", "plt.figure",
    "ax.", "fig.", "imshow", ".plot(", ".scatter(", ".hist(", ".bar(",
    "DataFrame", "summary()", "stats()", "pprint", "tabulate",
)


def _produces_output(src: str) -> bool:
    code = "".join(line for line in src.splitlines()
                   if not line.lstrip().startswith("#"))
    if any(c in code for c in _OUTPUT_CUES):
        return True
    last = code.rstrip().splitlines()[-1] if code.strip() else ""
    if last and not (
        last.startswith(("import ", "from ", "assert ", " "))
        or ("=" in last and "==" not in last and "!=" not in last)
    ):
        return True
    return False


def _all_code_cells_have_outputs(nb: dict) -> tuple[bool, str]:
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        if not _produces_output(src):
            continue
        outs = cell.get("outputs", [])
        if not outs:
            return False, f"code cell #{i} has no outputs"
        for o in outs:
            if o.get("output_type") == "error":
                return False, f"code cell #{i} errored: {o.get('evalue', '?')}"
    return True, "ok"


def _flatten_outputs(nb: dict) -> str:
    """Concatenate stream/text outputs from every code cell for spot-checks."""
    parts: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for o in cell.get("outputs", []):
            if "text" in o:
                parts.append("".join(o["text"]) if isinstance(o["text"], list)
                             else o["text"])
            elif o.get("output_type") == "execute_result":
                data = o.get("data", {})
                if "text/plain" in data:
                    val = data["text/plain"]
                    parts.append("".join(val) if isinstance(val, list) else val)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Per-notebook spot-checks
# ---------------------------------------------------------------------------

def _spot_check_nb_00(out: str) -> tuple[bool, str]:
    if "typed verdicts" not in out:
        return False, "expected 'typed verdicts' marker not found"
    if "category" not in out:
        return False, "expected per-category breakdown not found"
    return True, "ok"


def _spot_check_nb_02(out: str) -> tuple[bool, str]:
    if "514" not in out or "823" not in out:
        return False, "expected ch04 stats (514 entities / 823 triples) not seen"
    return True, "ok"


def _spot_check_nb_03(out: str) -> tuple[bool, str]:
    if "wall time" not in out or "accuracy" not in out:
        return False, "expected wall-time and accuracy markers not seen"
    return True, "ok"


def _spot_check_nb_19(out: str) -> tuple[bool, str]:
    for tier in ("advisory", "recommendation", "action_taking"):
        if tier not in out:
            return False, f"missing tier '{tier}' in release-gate output"
    return True, "ok"


def _spot_check_nb_20(out: str) -> tuple[bool, str]:
    if "ECE" not in out:
        return False, "expected ECE marker not found"
    return True, "ok"


def _spot_check_nb_21(out: str) -> tuple[bool, str]:
    if "passed:" not in out and "/9" not in out and "9 checks" not in out:
        return False, "expected diagnostic pass count not found"
    return True, "ok"


def _spot_check_nb_22(out: str) -> tuple[bool, str]:
    if "FailureCode members" not in out and "29" not in out:
        return False, "expected FailureCode taxonomy marker not found"
    return True, "ok"


SPOT_CHECKS: dict[str, callable] = {
    "00": _spot_check_nb_00,
    "02": _spot_check_nb_02,
    "03": _spot_check_nb_03,
    "19": _spot_check_nb_19,
    "20": _spot_check_nb_20,
    "21": _spot_check_nb_21,
    "22": _spot_check_nb_22,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def validate_notebook(nb_path: Path) -> tuple[bool, str]:
    nb_path = Path(nb_path)
    if not nb_path.exists():
        return False, f"notebook not found: {nb_path}"
    with open(nb_path) as f:
        nb = json.load(f)

    ok, reason = _all_code_cells_have_outputs(nb)
    if not ok:
        return False, reason

    m = NB_RE.match(nb_path.name)
    if m:
        prefix = m.group(1)
        spot = SPOT_CHECKS.get(prefix)
        if spot:
            flat = _flatten_outputs(nb)
            ok, reason = spot(flat)
            if not ok:
                return False, f"spot-check failed: {reason}"
    return True, "ok"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None,
                        help="validate only the NN-prefixed notebook")
    args = parser.parse_args(argv)

    rows = []
    for p in sorted(NOTEBOOKS_DIR.iterdir()):
        m = NB_RE.match(p.name)
        if not m:
            continue
        if args.only and m.group(1) != args.only.zfill(2):
            continue
        ok, reason = validate_notebook(p)
        rows.append((p.name, ok, reason))

    if not rows:
        print("[validate] no notebooks matched")
        return 0

    width = max(len(r[0]) for r in rows)
    print(f"{'notebook':<{width}}  status  reason")
    for name, ok, reason in rows:
        status = "PASS" if ok else "FAIL"
        print(f"{name:<{width}}  {status:<6}  {reason}")

    failed = sum(1 for r in rows if not r[1])
    print(f"\n[validate] {len(rows) - failed}/{len(rows)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
