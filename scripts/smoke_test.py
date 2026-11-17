#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
""" repo-local smoke-test entry point.

Thin wrapper over :func:`harness._smoke.main` so the editable-checkout
CI path (``packaging/verify_wheels.sh``, ``.github/workflows/ci.yml``)
and the installed ``gmsh-smoke`` console script share a single code
path.  All implementation lives in ``harness/_smoke.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from an editable checkout without `pip install -e` —
# the repo root carries `harness/` as a top-level package.  An
# installed wheel resolves `harness` through site-packages instead,
# so this path injection is a no-op in that path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from knowlytix.harness._smoke import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
