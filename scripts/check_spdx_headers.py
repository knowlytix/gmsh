#!/usr/bin/env python3
"""GMS-106: CI-blocking SPDX-License-Identifier header lint.

Runs in the ``spdx-lint`` job on every PR and push to
``main``/``staging``/``release``. Blocks on any of:

* Missing SPDX tag in a shipped-source ``.py`` file.
* Wrong identifier for the file's tree (e.g. ``Apache-2.0`` on a
  ``gms/`` file, or the EULA identifier on a framework file).
* Wrong placement (tag not on line 1, or not on line 2 immediately
  after a shebang — no blank line between shebang and tag).

Test and helper trees (``tests/``, ``scripts/``) are exempt per
GMS-030 rule 3. Vendored third-party files in
``_spdx_common.VENDORED_FILES`` are exempt per rule 6.

When a violation fires, the output names the file, line, expected
identifier, and suggests ``scripts/backfill_spdx_headers.py --apply``
for the common missing-header case.

Exit codes:

* ``0`` — every shipped ``.py`` file passes.
* ``1`` — one or more violations; stderr lists them.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _spdx_common import (  # noqa: E402
    REPO_ROOT,
    SPDX_TAG_PREFIX,
    expected_spdx_for,
    iter_shipped_py_files,
    parse_header,
)


def main() -> int:
    violations: list[str] = []
    paths = iter_shipped_py_files()

    for path in paths:
        expected = expected_spdx_for(path)
        assert expected is not None, f"no SPDX mapping for {path}"
        info = parse_header(path.read_text())
        rel = path.relative_to(REPO_ROOT).as_posix()

        if info.spdx_value is None:
            violations.append(
                f"{rel}: missing SPDX header "
                f"(expected '{SPDX_TAG_PREFIX}{expected}')"
            )
            continue
        if info.spdx_value != expected:
            violations.append(
                f"{rel}:{info.spdx_line}: wrong SPDX identifier "
                f"(have {info.spdx_value!r}, want {expected!r})"
            )
            continue
        if not info.correct_placement:
            expected_line = 2 if info.has_shebang else 1
            violations.append(
                f"{rel}:{info.spdx_line}: SPDX tag on wrong line "
                f"(expected line {expected_line}; no blank line between "
                "shebang and SPDX tag)"
            )

    if violations:
        print("SPDX lint FAILED:", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        print(
            f"\n{len(violations)} violation(s). For the common "
            "missing-header case, run:\n"
            "  python scripts/backfill_spdx_headers.py --apply\n"
            "Other violations require manual review.",
            file=sys.stderr,
        )
        return 1

    print(f"SPDX lint OK: {len(paths)} shipped .py files scanned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
