# SPDX-License-Identifier: Apache-2.0
"""Package marker for ``harness.smoke`` — resource-discovery root.

Enables ``importlib.resources.files("harness.smoke")`` for the
``knowlytix-smoke --notebook-path`` flag.  Ships ``smoke.ipynb`` (the
human-interactive companion to ``harness._smoke``'s 5-step assertion
suite) as package data.

No module-level code — this file exists only to make the package
importable for resource discovery, not to expose any public API.
"""
