"""Shared helpers for the SPDX tooling.

Used by ``scripts/backfill_spdx_headers.py`` (one-shot header insertion)
and ``scripts/check_spdx_headers.py`` (CI-blocking lint). Both scripts
prepend this file's directory to ``sys.path`` and import the symbols
they need — keeping the walker, the placement parser, and the
per-tree identifier map in one place so backfill and lint can never
drift against each other.

Convention is defined by (see
``distribution/release_readiness_plan.md`` §1). Quick recap:

* Every shipped-source ``.py`` file carries
  ``# SPDX-License-Identifier: <spdx>`` as the first line — or line 2
  immediately after a shebang (no blank line between).
* Per-tree identifiers: ``knowlytix/core/`` → commercial EULA;
  ``knowlytix/doc/``, ``knowlytix/bench/`` (including
  ``knowlytix/bench/eval/``), ``knowlytix/harness/``, and
  ``knowlytix/governed_agent/`` (Phase 4.10) → Apache-2.0.
* ``tests/`` and ``scripts/`` are exempt; vendored third-party files
  are listed in ``VENDORED_FILES`` and skipped verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Per- rule 3: shipped tree → SPDX identifier.
SPDX_BY_TREE: dict[str, str] = {
    # gms/ moved to knowlytix/core/; legacy gms/ shim stays
    # for compat until removes it.
    "knowlytix/core": "LicenseRef-Knowlytix-GMS-EULA",
    # knowlytix/embedding is commercial (low-rank SFT); it rides in the
    # knowlytix-core wheel Nuitka-compiled, same EULA tier as core.
    "knowlytix/embedding": "LicenseRef-Knowlytix-GMS-EULA",
    "gms": "LicenseRef-Knowlytix-GMS-EULA",
    # docgms/ -> knowlytix/doc/ -> knowlytix/knowledge/; the
    # benchmark tree settled at knowlytix/benchmark/. The old doc/bench keys
    # matched nothing post-rename, silently leaving both trees unscanned.
    "knowlytix/knowledge": "Apache-2.0",
    "knowlytix/benchmark": "Apache-2.0",
    "knowlytix/harness": "Apache-2.0",
    # (Phase 4.10): governed_agent/ moved to knowlytix/governed_agent/
    # as the foundation for shipping knowlytix-governed-agent as the 5th
    # Apache-2.0 wheel at v0.2.0. backfill SPDX headers across
    # the 131 .py files in the moved tree.
    "knowlytix/governed_agent": "Apache-2.0",
}

# Further trees listed beside the commercial packaging ("<tree> <SPDX id>" per line).
_EXTRA_TREES = REPO_ROOT / "packaging" / "knowlytix-core" / "spdx_trees.txt"
if _EXTRA_TREES.is_file():
    for _line in _EXTRA_TREES.read_text().splitlines():
        if _line.strip() and not _line.lstrip().startswith("#"):
            _tree, _spdx = _line.split()
            SPDX_BY_TREE[_tree] = _spdx

# Top-level directories that ship in wheels. Anything outside these is
# out of scope for the lint (repo-private tooling, book content, etc.).
SHIPPED_DIRS: tuple[str, ...] = tuple(SPDX_BY_TREE)

# Directory-name components that exempt a file from the lint.
# ``tests/`` covers every level (e.g. ``harness/doe/tests/*``); the
# others are compile/cache artifacts that shouldn't exist in a clean
# checkout but are filtered defensively.
EXEMPT_DIR_PARTS: frozenset[str] = frozenset({
    "__pycache__",
    ".pytest_cache",
    "tests",
})

# Vendored third-party files keep their upstream license header
# verbatim per rule 6. Populate with repo-relative POSIX paths
# as vendored code lands; the lint consults this allowlist and the
# backfill never overwrites entries here.
VENDORED_FILES: frozenset[str] = frozenset({
    # e.g. "harness/doe/thirdparty/<upstream>/foo.py"
})

SPDX_TAG_PREFIX = "# SPDX-License-Identifier: "


@dataclass(frozen=True)
class HeaderInfo:
    """Result of parsing the first two lines of a ``.py`` file."""

    has_shebang: bool
    spdx_line: int | None  # 1-indexed; None if no SPDX tag present
    spdx_value: str | None  # identifier string, without the prefix
    correct_placement: bool  # True iff placement obeys rule 1


def tree_for(path: Path) -> str | None:
    """Return the shipped top-level tree that owns ``path``, or None.

    Handles both single-level trees (``gms``, ``harness``) and
    two-level namespace trees (``knowlytix/core``, ``knowlytix/doc``).
    """
    try:
        rel = path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return None
    parts = rel.parts
    if not parts:
        return None
    # Try two-level match first (e.g. knowlytix/core).
    if len(parts) >= 2:
        two_level = f"{parts[0]}/{parts[1]}"
        if two_level in SHIPPED_DIRS:
            return two_level
    head = parts[0]
    return head if head in SHIPPED_DIRS else None


def expected_spdx_for(path: Path) -> str | None:
    """Return the SPDX identifier this path must carry, or None if out of scope."""
    tree = tree_for(path)
    return SPDX_BY_TREE.get(tree) if tree else None


def parse_header(text: str) -> HeaderInfo:
    """Inspect the first two lines of ``text`` for SPDX shape."""
    # Only the first two lines can hold a valid SPDX header per rule 1.
    # Read at most that many — the rest of the file is irrelevant here.
    first_two = text.split("\n", 2)[:2]
    has_shebang = bool(first_two) and first_two[0].startswith("#!")
    expected_idx = 1 if has_shebang else 0  # 0-indexed target line

    spdx_line: int | None = None
    spdx_value: str | None = None
    for idx, line in enumerate(first_two):
        stripped = line.rstrip()
        if stripped.startswith(SPDX_TAG_PREFIX):
            spdx_line = idx + 1  # 1-indexed for human-readable errors
            spdx_value = stripped[len(SPDX_TAG_PREFIX):].strip()
            break

    correct_placement = spdx_line is not None and (spdx_line - 1) == expected_idx
    return HeaderInfo(
        has_shebang=has_shebang,
        spdx_line=spdx_line,
        spdx_value=spdx_value,
        correct_placement=correct_placement,
    )


def iter_shipped_py_files() -> list[Path]:
    """Yield absolute paths to every shipped-source ``.py`` file in scope.

    Applies rule 3 (tests/scripts exempt) and rule 6 (vendored
    allowlist). Output order is stable across runs so diffs stay
    reviewable.
    """
    out: list[Path] = []
    for top in SHIPPED_DIRS:
        root = REPO_ROOT / top
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if any(part in EXEMPT_DIR_PARTS for part in path.parts):
                continue
            rel_posix = path.relative_to(REPO_ROOT).as_posix()
            if rel_posix in VENDORED_FILES:
                continue
            out.append(path)
    return out
