# SPDX-License-Identifier: Apache-2.0
"""Repo-root bootstrap: clear the license gate for every collected suite.

``knowlytix.core`` is license-gated, so any test that imports it — directly or
transitively — needs ``KNOWLYTIX_LICENSE_KEY`` set before the import happens.
``tests/conftest.py`` has always done this, but a conftest only applies to its
own directory and below, and ``testpaths`` now reaches well outside ``tests/``
(e.g. knowlytix/knowledge/geode/tests, knowlytix/harness/testing/tests).

Those out-of-tree suites were clearing the gate only because ``tests`` sorts
first in ``testpaths``: its conftest is imported during collection, sets the
env var process-wide, and everything collected afterwards inherits it. That
works, but it is an ordering accident, not a contract — running a suite on its
own skips it entirely: running one of those suites by itself fails with
``DemoModeError`` on any machine without a personal ``~/.knowlytix/license.key``.

Living at the repo root, this file is imported before *any* testpath is
collected, so the guarantee holds however the suite is invoked.

Deliberately narrow: only the license/EULA bootstrap lives here, because that
is the one thing every suite needs. Fixtures that only ``tests/`` requires —
the offline text-encoder stub, and so on — stay in ``tests/conftest.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_PRIVATE_KEY_PATH = _REPO_ROOT / "knowlytix-private.pem"


def _bootstrap_license_key() -> None:
    """Set ``KNOWLYTIX_LICENSE_KEY`` if missing and the private key is present.

    No-ops when the variable is already set (a real license, or CI's ephemeral
    keypair) and when the key is absent, which is the case in a fresh clone. In
    that second case the gated tests still fail — but with the gate's own
    actionable message rather than a confusing import error, which is the same
    trade-off ``tests/conftest.py`` already makes.
    """
    if os.environ.get("KNOWLYTIX_LICENSE_KEY"):
        return
    if not _PRIVATE_KEY_PATH.is_file():
        return

    # Imported lazily: `tests` is only importable once the repo root is on
    # sys.path, which pytest arranges by the time conftest runs, and a fresh
    # clone without the private key never needs it at all.
    from tests._helpers import make_bootstrap_token

    os.environ["KNOWLYTIX_LICENSE_KEY"] = make_bootstrap_token(
        _PRIVATE_KEY_PATH.read_text(encoding="utf-8"),
        customer_id="pytest-repo-root-bootstrap",
    )


# Suppress the EULA reminder for the whole session so collection-time imports
# of `knowlytix.core` don't spam stderr. Tests that exercise the banner itself
# unset this in their own monkeypatch setup.
os.environ.setdefault("KNOWLYTIX_EULA_ACCEPTED", "1")

_bootstrap_license_key()
