#!/usr/bin/env python3
""" one-shot SPDX-License-Identifier header insertion.

Walks every shipped-source ``.py`` file (``knowlytix/core/``,
``knowlytix/doc/``, ``gms/`` compat shim, ``knowlytix/bench/`` —
including ``knowlytix/bench/eval/`` — and ``knowlytix/harness/``) and inserts the
correct SPDX tag per's per-tree mapping:

* ``knowlytix/core/**/*.py``, ``gms/**/*.py`` → ``# SPDX-License-Identifier: LicenseRef-Knowlytix-GMS-EULA``
* ``knowlytix/doc/**/*.py``, ``knowlytix/bench/**/*.py``, ``knowlytix/harness/**/*.py``
  → ``# SPDX-License-Identifier: Apache-2.0``

Placement follows rule 1: first line if no shebang, else line
2 immediately after the shebang (no blank line between).

Skip rules:

* ``tests/`` and ``scripts/`` are exempt per rule 3.
* Vendored files in ``_spdx_common.VENDORED_FILES`` are left alone per
  rule 6.
* Files that already carry a matching header in the correct placement
  are no-ops.
* Files that carry a *different* SPDX identifier, or the correct
  identifier in the wrong placement, are flagged as conflicts and left
  alone — operator decides whether to fix by hand. Non-zero exit makes
  those visible in CI.

Usage::

    python scripts/backfill_spdx_headers.py           # dry-run (default)
    python scripts/backfill_spdx_headers.py --apply   # write changes

The dry-run path prints the file list and counts so you can review
before writing. Run once, commit the diff in a single PR titled
"SPDX: backfill license headers per" per the story AC.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `from _spdx_common import ...` when this script is invoked from
# anywhere. ``scripts/`` isn't a package, so the usual relative import
# won't resolve; prepending the script's own directory is the idiomatic
# escape hatch.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _spdx_common import (  # noqa: E402
    SPDX_TAG_PREFIX,
    expected_spdx_for,
    iter_shipped_py_files,
    parse_header,
)


def insert_header(text: str, spdx: str) -> str:
    """Return ``text`` with an SPDX header inserted at the correct line."""
    tag = f"{SPDX_TAG_PREFIX}{spdx}"
    # Split preserves all trailing newlines exactly; re-joining with
    # the same separator round-trips the file content byte-for-byte
    # except for the new header line (plus its trailing newline).
    lines = text.split("\n")
    if lines and lines[0].startswith("#!"):
        # Preserve the shebang as line 1; SPDX becomes line 2.
        return "\n".join([lines[0], tag, *lines[1:]])
    # No shebang — SPDX becomes the new first line.
    return "\n".join([tag, *lines])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=" SPDX backfill.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes in place (default: dry-run, no mutation).",
    )
    args = parser.parse_args(argv)

    would_change: list[Path] = []
    conflicts: list[tuple[Path, str]] = []
    already_ok: list[Path] = []

    for path in iter_shipped_py_files():
        expected = expected_spdx_for(path)
        # iter_shipped_py_files only yields paths with a mapping; the
        # assert guards against a future refactor silently expanding
        # the walker without updating SPDX_BY_TREE.
        assert expected is not None, f"no SPDX mapping for {path}"

        text = path.read_text()
        info = parse_header(text)

        if info.spdx_value is not None:
            if info.spdx_value == expected and info.correct_placement:
                already_ok.append(path)
                continue
            # Existing header either disagrees with the expected
            # identifier or sits on the wrong line — operator action.
            reason = (
                f"have={info.spdx_value!r} want={expected!r} "
                f"placement_ok={info.correct_placement}"
            )
            conflicts.append((path, reason))
            continue

        # No header yet — candidate for insertion.
        new_text = insert_header(text, expected)
        if args.apply:
            path.write_text(new_text)
        would_change.append(path)

    total = len(already_ok) + len(would_change) + len(conflicts)
    print(f"SPDX backfill summary (total shipped .py files: {total})")
    print(f"  already OK:        {len(already_ok)}")
    print(f"  {'wrote' if args.apply else 'would insert'}: {len(would_change)}")
    print(f"  conflicts:         {len(conflicts)}")

    if conflicts:
        print("\nConflicts require manual review:")
        for path, reason in conflicts:
            rel = path.relative_to(Path.cwd()) if path.is_absolute() else path
            print(f"  {rel}: {reason}")

    if not args.apply and would_change:
        print(
            "\n(dry-run; re-run with --apply to write "
            f"{len(would_change)} file(s))"
        )

    return 1 if conflicts else 0


if __name__ == "__main__":
    sys.exit(main())
