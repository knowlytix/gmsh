"""Shared pytest fixtures for tests that need a real ingested GMS store.

The canonical test store is ``monograph/lab_stores/ch04_ground_truth`` —
a 4-run, 200-epoch ingest of ``model_risk_assessment.md``.  These
fixtures load it once per session (no re-ingestion) and skip the
affected tests cleanly if it is missing, so the suite degrades to a
useful subset on a fresh clone.
"""

from __future__ import annotations

import os

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          "..", "..", ".."))
CH04_STORE_PATH = os.path.join(REPO_ROOT, "monograph", "lab_stores",
                               "ch04_ground_truth")


@pytest.fixture(scope="session")
def ch04_store():
    """Load the real ch04_ground_truth GMSExpertStore from disk.

    Skips the test if the store has not been ingested.  Loads once per
    pytest session — downstream tests only call geometric primitives
    (``score_triple``, ``tension_energy``, ``fuzzy_match_entity``), all
    of which are cheap after load.
    """
    from knowlytix.knowledge.config import DocGMSConfig
    from knowlytix.knowledge.store import GMSExpertStore

    if not os.path.isfile(os.path.join(CH04_STORE_PATH, "model.pt")):
        pytest.skip(
            f"real store missing at {CH04_STORE_PATH}; "
            f"run the Ch4 ingest to produce it"
        )

    config = DocGMSConfig(store_path=CH04_STORE_PATH)
    store = GMSExpertStore(config)
    assert store.load(), "GMSExpertStore.load() returned False"
    assert store.doc_graph is not None, "doc_graph did not load"
    assert len(list(store.doc_graph.triples)) > 0, "store has no triples"
    return store


@pytest.fixture
def mock_environment():
    """Fresh EnvironmentSimulator for each test.

    Per-test scope so ticket counters / query counts don't leak
    between tests.
    """
    from demo.mock_agent_full import create_mock_environment
    return create_mock_environment()


@pytest.fixture
def full_mock_harness(mock_environment):
    """FullMockHarness wired to a fresh mock environment."""
    from demo.mock_agent_full import FullMockHarness
    return FullMockHarness(
        environment=mock_environment, recovery_mode="fallback",
    )
