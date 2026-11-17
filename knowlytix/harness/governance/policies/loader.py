# SPDX-License-Identifier: Apache-2.0
"""Policy YAML loader for governed agents.

Loads YAML policy rules into :class:`~knowlytix.harness.testing.policy.PolicyRule`
objects and optionally ingests them as GMS triples into a store.

Usage::

    from knowlytix.harness.governance.policies.loader import load_policy_yaml, rules_to_triples

    rules = load_policy_yaml("governed_agent/policies/responsible_ml.yaml")
    triples = rules_to_triples(rules)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from knowlytix.harness.testing.policy import PolicyRule
from knowlytix.knowledge.extract import ExtractedTriple

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore


def load_policy_yaml(path: str) -> list[PolicyRule]:
    """Load policy rules from a YAML file.

    Parameters
    ----------
    path : str
        Path to the YAML file.  Must contain a top-level ``rules`` key.

    Returns
    -------
    list[PolicyRule]
        Deserialized policy rules.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file lacks a ``rules`` key.
    """
    with open(path, "r") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict) or "rules" not in data:
        raise ValueError(
            f"YAML file must contain a top-level 'rules' key: {path}"
        )

    return [PolicyRule.from_dict(entry) for entry in data["rules"]]


def rules_to_triples(rules: list[PolicyRule]) -> list[ExtractedTriple]:
    """Convert policy rules to GMS triples for ingestion.

    For each rule the following triples are produced:

    - ``(action, rule_type, requirement)`` -- the primary policy triple
    - ``(rule_id, "is_type", "policy_rule")`` -- metadata
    - ``(rule_id, "applies_to", action)`` -- metadata
    - ``(rule_id, "severity", severity)`` -- metadata

    Parameters
    ----------
    rules : list[PolicyRule]
        Rules to convert.

    Returns
    -------
    list[ExtractedTriple]
        GMS-ready triples.
    """
    triples: list[ExtractedTriple] = []
    for rule in rules:
        triples.append(ExtractedTriple(
            head=rule.action,
            relation=rule.rule_type,
            tail=rule.requirement,
        ))
        triples.append(ExtractedTriple(
            head=rule.rule_id,
            relation="is_type",
            tail="policy_rule",
        ))
        triples.append(ExtractedTriple(
            head=rule.rule_id,
            relation="applies_to",
            tail=rule.action,
        ))
        triples.append(ExtractedTriple(
            head=rule.rule_id,
            relation="severity",
            tail=rule.severity,
        ))
    return triples


def ingest_rules_to_store(
    rules: list[PolicyRule],
    store: "GMSExpertStore",
) -> int:
    """Ingest policy rules as GMS triples into a store.

    Uses :class:`~knowlytix.knowledge.learn.RuntimeLearner` to encode the triples
    into the store's geometric model.

    Parameters
    ----------
    rules : list[PolicyRule]
        Rules to ingest.
    store : GMSExpertStore
        Target store.

    Returns
    -------
    int
        Number of triples added.
    """
    import torch

    from knowlytix.knowledge.config import LearnConfig
    from knowlytix.knowledge.learn import RuntimeLearner

    triples = rules_to_triples(rules)

    if hasattr(store, "config") and store.config is not None:
        learn_cfg = store.config.learn
    else:
        learn_cfg = LearnConfig()

    learner = RuntimeLearner(store, learn_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = learner.learn_triples(triples, device)
    return result.n_triples_added
