# SPDX-License-Identifier: Apache-2.0
"""GMS-200 spike — execute tutorial notebooks against an active venv, log results.

Categorizes every `*.ipynb` in `gmsh_testing_ext/notebooks/` (excluding
`*_executed.ipynb` archival copies) by dependency class, then runs the
unblocked ones end-to-end via `jupyter nbconvert --to notebook --execute`.

Categories (from import + API-usage scan):
    llm          — `import anthropic`, OR constructs `DOEGMSBenchmark`
                    (whose `.ingest()` uses `AnthropicBackend()` for
                    triple extraction). Requires `ANTHROPIC_API_KEY`
                    in the process env — document `.env` + `source`
                    pattern in `docs/internal_team_install.md` §3B.
    clean        — no known external dep beyond knowlytix.* (runnable
                    without an API key).

`graphdoe` category was retired 2026-04-21 after the upstream vendor
drop (sync/upstream-YYYY-MM-DD branch cherry-picked
`09a4151 Vendor graphdoe…`); see `distribution/tutorials_port_log.md`
for historical context.

`--only clean` runs only the clean set (default).
`--only all` tries everything; llm notebooks fail explicitly without
a key.

`--notebook-dir` points at the directory to scan. Default
`gmsh_testing_ext/notebooks` matches the monorepo layout; mirror CI
(GMS-172 post-mirror-smoke) passes `tutorials/notebooks` instead.

Usage:
    # Prereqs: gmsh clone venv activated; PYTHONPATH=<gmsh-clone-root>;
    # ipykernel registered (e.g. `python -m ipykernel install --user --name knowlytix`).
    python3 scripts/execute_tutorials.py --kernel knowlytix-ch02
    python3 scripts/execute_tutorials.py --kernel knowlytix-ch02 --only clean
    python3 scripts/execute_tutorials.py --kernel knowlytix-ch02 --log-path distribution/tutorials_port_log.md
    python3 scripts/execute_tutorials.py --kernel gmsh-ci --only clean \\
        --notebook-dir tutorials/notebooks --log-path /tmp/tutorials_smoke.md
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

NOTEBOOK_DIR_DEFAULT = Path("gmsh_testing_ext/notebooks")
LOG_PATH_DEFAULT = Path("distribution/tutorials_port_log.md")

Category = Literal["clean", "llm"]


def _scan_category(nb_path: Path) -> tuple[Category, list[str]]:
    """Classify a notebook by its import + API-usage signatures."""
    text = nb_path.read_text()
    nb = json.loads(text)
    signals: list[str] = []

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if re.search(r"^\s*import anthropic\b", src, re.MULTILINE):
            signals.append("anthropic")
        if re.search(r"^\s*from graphdoe\b", src, re.MULTILINE):
            signals.append("graphdoe-import")
        if re.search(r"DOEGMSBenchmark\s*\(", src):
            signals.append("DOEGMSBenchmark-use")

    # `DOEGMSBenchmark(...).ingest()` constructs `AnthropicBackend()`
    # for LLM-driven triple extraction from the source markdown.
    # Treat any DOEGMSBenchmark use as LLM-dependent even if the
    # notebook never directly imports anthropic.
    if signals and ("anthropic" in signals or "DOEGMSBenchmark-use" in signals):
        return "llm", signals
    return "clean", signals


def _execute(
    nb_path: Path,
    kernel: str,
    timeout: int,
    out_dir: Path,
) -> tuple[bool, str, float]:
    """Run `jupyter nbconvert --execute`. Returns (passed, detail, seconds)."""
    out_path = out_dir / f"{nb_path.stem}_spike.ipynb"
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        f"--ExecutePreprocessor.kernel_name={kernel}",
        f"--ExecutePreprocessor.timeout={timeout}",
        "--output", str(out_path),
        str(nb_path.resolve()),
    ]
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout + 30,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s", time.perf_counter() - t0
    dt = time.perf_counter() - t0

    if result.returncode != 0:
        err_tail = (result.stderr or "").splitlines()[-6:]
        return False, "\n    ".join(err_tail) or "nbconvert non-zero", dt

    # Double-check: even on success, scan output notebook for cell errors
    # (some errors are silently recoverable in convert but show in outputs).
    if not out_path.is_file():
        return False, "output notebook missing", dt
    out_nb = json.loads(out_path.read_text())
    for cell in out_nb["cells"]:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                name = output.get("ename", "?")
                value = (output.get("evalue") or "").splitlines()[0][:120]
                return False, f"{name}: {value}", dt
    return True, "", dt


def _render_log(
    results: list[dict], only: str, kernel: str, venv: str | None
) -> str:
    """Render the per-notebook log in markdown."""
    now = time.strftime("%Y-%m-%d")
    lines = [
        "# Tutorials port log — GMS-200 spike",
        "",
        f"**Run date**: {now}",
        f"**Filter**: `--only {only}`",
        f"**Kernel**: `{kernel}`",
        f"**Venv**: `{venv or '(inherited)'}`",
        "",
        "Generated by `scripts/execute_tutorials.py`.",
        "",
        "| Notebook | Category | Status | Time (s) | Detail |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        status = (
            "✅ pass"
            if r["passed"]
            else "⚠️ skipped"
            if r["status"] == "skipped"
            else "❌ fail"
        )
        detail = r["detail"].replace("|", "\\|").replace("\n", " · ")
        if len(detail) > 140:
            detail = detail[:137] + "..."
        lines.append(
            f"| `{r['name']}` | {r['category']} | {status} | "
            f"{r['seconds']:.1f} | {detail} |"
        )
    n_pass = sum(1 for r in results if r["passed"])
    n_skip = sum(1 for r in results if r["status"] == "skipped")
    n_fail = sum(1 for r in results if not r["passed"] and r["status"] != "skipped")
    lines.extend([
        "",
        f"**Summary**: {n_pass} passed, {n_fail} failed, {n_skip} skipped "
        f"({len(results)} total).",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel", required=True, help="Jupyter kernel name to execute against.",
    )
    parser.add_argument(
        "--only", choices=["clean", "all"], default="clean",
        help="`clean` skips llm+graphdoe notebooks; `all` runs everything.",
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Per-notebook execution timeout (seconds).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("/tmp/tutorials_spike_output"),
        help="Where to write executed notebook outputs.",
    )
    parser.add_argument(
        "--log-path", type=Path, default=LOG_PATH_DEFAULT,
        help="Markdown log destination.",
    )
    parser.add_argument(
        "--notebook-dir", type=Path, default=NOTEBOOK_DIR_DEFAULT,
        help=(
            "Directory to scan for *.ipynb files. Default matches the "
            "monorepo layout; mirror CI overrides to tutorials/notebooks."
        ),
    )
    args = parser.parse_args()

    if not args.notebook_dir.is_dir():
        print(
            f"error: {args.notebook_dir} not found (pass --notebook-dir or "
            f"run from repo root)",
            file=sys.stderr,
        )
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Only base notebooks (skip `*_executed.ipynb` archival copies).
    targets = sorted(
        p for p in args.notebook_dir.glob("*.ipynb")
        if not p.stem.endswith("_executed")
    )

    results: list[dict] = []
    for nb_path in targets:
        category, signals = _scan_category(nb_path)
        entry = {
            "name": nb_path.name,
            "category": category,
            "passed": False,
            "status": "",
            "detail": "",
            "seconds": 0.0,
        }
        if args.only == "clean" and category != "clean":
            entry["status"] = "skipped"
            entry["detail"] = f"category={category} signals={signals}"
            results.append(entry)
            print(f"SKIP   {nb_path.name} (category={category})")
            continue

        print(f"RUN    {nb_path.name} ...", flush=True)
        passed, detail, seconds = _execute(
            nb_path, args.kernel, args.timeout, args.out_dir,
        )
        entry["passed"] = passed
        entry["status"] = "pass" if passed else "fail"
        entry["detail"] = detail
        entry["seconds"] = seconds
        mark = "PASS  " if passed else "FAIL  "
        print(f"  {mark}{nb_path.name} ({seconds:.1f}s)"
              + (f"  — {detail.splitlines()[0][:80]}" if detail else ""))
        results.append(entry)

    log_text = _render_log(results, args.only, args.kernel, sys.prefix)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.write_text(log_text)

    n_pass = sum(1 for r in results if r["passed"])
    n_fail = sum(1 for r in results if not r["passed"] and r["status"] != "skipped")
    n_skip = sum(1 for r in results if r["status"] == "skipped")
    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    print(f"log written to {args.log_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
