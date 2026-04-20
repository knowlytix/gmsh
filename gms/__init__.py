# SPDX-License-Identifier: LicenseRef-Knowlytix-GMS-EULA
"""Temporary compat shim — GMS-112 moved ``gms/`` → ``knowlytix/core/``.

Consumers that still ``from gms import X`` or ``from gms.X.Y import Z``
(docgms, finstructbench, harness, provisioning, scripts, tests) are
bridged here until GMS-116 flips all remaining imports and removes this
file.

**Module-identity invariant**: every ``gms.*`` import MUST resolve to
the same ``sys.modules`` object as the corresponding
``knowlytix.core.*`` import.  Without this, module-level globals
(e.g. ``_caps._RESOLVED_CAPS`` set by ``verify_license()``) are
visible to one import path but not the other — causing fail-closed
"license not verified" errors despite the gate having run.  The
``_GmsAliasFinder`` meta-path hook below registers ``gms.*`` aliases
in ``sys.modules`` at import time to enforce identity.
"""

from __future__ import annotations

import importlib
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec


class _GmsAliasFinder(MetaPathFinder):
    """Redirect any ``gms.<submodule>`` import to ``knowlytix.core.<submodule>``.

    Installed once in ``sys.meta_path`` by this shim.  Removed when
    GMS-116 deletes the ``gms/`` compat package entirely.
    """

    _PREFIX = "gms."
    _TARGET = "knowlytix.core."

    def find_module(  # type: ignore[override]  # deprecated but still used
        self,
        fullname: str,
        path: object = None,
    ) -> _GmsAliasFinder | None:
        if fullname.startswith(self._PREFIX):
            return self
        return None

    def load_module(self, fullname: str) -> object:  # type: ignore[override]
        if fullname in sys.modules:
            return sys.modules[fullname]
        real = self._TARGET + fullname[len(self._PREFIX) :]
        mod = importlib.import_module(real)
        sys.modules[fullname] = mod
        return mod


# Install the finder BEFORE importing knowlytix.core — this way any
# submodule that `knowlytix.core.__init__` imports (e.g. `_license`,
# `_caps`) is immediately aliased under `gms.*` in sys.modules.
if not any(isinstance(f, _GmsAliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _GmsAliasFinder())

import knowlytix.core as _kc  # noqa: E402

# Re-export the full public surface.
from knowlytix.core import *  # noqa: F401,F403,E402

# Preserve __all__ and __version__ for test assertions (GMS-036).
__all__ = _kc.__all__  # type: ignore[attr-defined]
__version__: str = _kc.__version__  # type: ignore[attr-defined]

# Subpackage aliasing via __path__ — lets `from gms.geometry.sphere
# import X` resolve via the knowlytix/core/ tree.
__path__ = _kc.__path__  # type: ignore[attr-defined]
