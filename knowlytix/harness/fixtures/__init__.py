# SPDX-License-Identifier: Apache-2.0
"""Shipped fixtures for knowlytix-harness.

Public API: deterministic mock agents + a pre-built EnvironmentSimulator.
Accessed via ``from knowlytix.harness.fixtures import MockHarness, …``.

Smoke-test assets (referenced via ``importlib.resources``) live alongside
at ``knowlytix.harness.fixtures.smoke``.
"""

from knowlytix.harness.fixtures.mock_agent import (
    MOCK_TOOLS,
    MockHarness,
    create_mock_evaluator,
)
from knowlytix.harness.fixtures.mock_agent_full import (
    ENVIRONMENT_TOOLS,
    FullMockHarness,
    MockStreamClient,
    create_agent_for_role,
    create_full_evaluator,
    create_mock_environment,
    create_mock_stream_client,
    create_multi_turn_script,
)

__all__ = [
    # From mock_agent.py
    "MOCK_TOOLS",
    "MockHarness",
    "create_mock_evaluator",
    # From mock_agent_full.py
    "ENVIRONMENT_TOOLS",
    "FullMockHarness",
    "MockStreamClient",
    "create_agent_for_role",
    "create_full_evaluator",
    "create_mock_environment",
    "create_mock_stream_client",
    "create_multi_turn_script",
]
