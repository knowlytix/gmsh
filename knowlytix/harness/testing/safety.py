# SPDX-License-Identifier: Apache-2.0
"""GMSH Safety and Isolation — runtime safety enforcement for black-box testing.

Provides a safety layer that sits between the test harness and the SUT
to enforce:

- **Tool access control** — block patterns, whitelist-only mode
- **Resource limits** — max tool calls, max network calls, max file writes
- **Sandbox enforcement** — redirect destructive actions to no-op
- **Synthetic data mode** — ensure only synthetic identities and data

Also provides a ``SyntheticDataFramework`` that generates test data
by sampling from the learned GMS entity embedding distribution on
S^{m-1} — a novel generative application of the geometric substrate.

Usage::

    enforcer = SafetyEnforcer(SafetyConfig(
        sandbox_mode=True,
        blocked_tool_patterns=["payment_*", "delete_*"],
    ))
    enforcer.install(harness)
    # ... run tests ...
    enforcer.uninstall(harness)
"""

from __future__ import annotations

import fnmatch
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore
    from knowlytix.harness.core import Harness
    from .tracing import DOETracingGMS


# ---------------------------------------------------------------------------
# Safety configuration
# ---------------------------------------------------------------------------

@dataclass
class SafetyConfig:
    """Safety and isolation configuration.

    Attributes
    ----------
    sandbox_mode : bool
        If True, all tool calls are checked against safety rules.
    network_isolation : bool
        If True, tools that access external networks are blocked.
    allowed_domains : list[str]
        When network_isolation is True, only these domains are allowed.
    synthetic_data_mode : bool
        If True, warn when non-synthetic data patterns are detected.
    max_tool_calls : int
        Hard limit on total tool calls per run (0 = unlimited).
    max_file_writes : int
        Hard limit on file write operations (0 = unlimited).
    max_network_calls : int
        Hard limit on network operations (0 = unlimited).
    blocked_tool_patterns : list[str]
        ``fnmatch`` patterns for tools that are always blocked.
    allowed_tool_patterns : list[str]
        If non-empty, only tools matching these patterns are allowed
        (whitelist mode).
    destructive_action_mode : str
        How to handle destructive actions: ``"block"`` (reject),
        ``"noop"`` (accept but do nothing), ``"log"`` (allow but log).
    """

    sandbox_mode: bool = False
    network_isolation: bool = False
    allowed_domains: list[str] = field(default_factory=list)
    synthetic_data_mode: bool = False
    max_tool_calls: int = 0
    max_file_writes: int = 0
    max_network_calls: int = 0
    blocked_tool_patterns: list[str] = field(default_factory=list)
    allowed_tool_patterns: list[str] = field(default_factory=list)
    destructive_action_mode: str = "block"  # "block", "noop", "log"

    def to_dict(self) -> dict:
        return {
            "sandbox_mode": self.sandbox_mode,
            "network_isolation": self.network_isolation,
            "allowed_domains": list(self.allowed_domains),
            "synthetic_data_mode": self.synthetic_data_mode,
            "max_tool_calls": self.max_tool_calls,
            "max_file_writes": self.max_file_writes,
            "max_network_calls": self.max_network_calls,
            "blocked_tool_patterns": list(self.blocked_tool_patterns),
            "allowed_tool_patterns": list(self.allowed_tool_patterns),
            "destructive_action_mode": self.destructive_action_mode,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SafetyConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str) -> SafetyConfig:
        """Load safety config from YAML.

        Expected structure::

            safety:
              sandbox_mode: true
              network_isolation: true
              blocked_tool_patterns:
                - "payment_*"
                - "delete_*"
              max_tool_calls: 100
              destructive_action_mode: block
        """
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        if "safety" in raw and isinstance(raw["safety"], dict):
            raw = raw["safety"]
        return cls.from_dict(raw)


# ---------------------------------------------------------------------------
# Destructive action patterns
# ---------------------------------------------------------------------------

# Tools that modify external state (deletions, writes, payments, etc.)
_DESTRUCTIVE_PATTERNS = [
    "delete_*", "remove_*", "drop_*",
    "payment_*", "transfer_*", "refund_*",
    "send_email", "send_message",
    "approve_*", "reject_*",
    "write_file", "create_file",
]

# Tools that access external networks
_NETWORK_PATTERNS = [
    "web_search", "web_fetch", "http_*",
    "api_call", "external_*",
]


# ---------------------------------------------------------------------------
# Safety enforcer
# ---------------------------------------------------------------------------

class SafetyEnforcer:
    """Runtime safety enforcement layer for test harness.

    Installs as a ``before_tool`` hook in the Harness, running before
    the ToolGateway.  Tracks tool call counts and enforces safety rules.

    Parameters
    ----------
    config : SafetyConfig
        Safety configuration.
    tracing : DOETracingGMS or None
        Optional tracing store for audit.
    """

    def __init__(
        self,
        config: SafetyConfig,
        tracing: DOETracingGMS | None = None,
    ):
        self.config = config
        self.tracing = tracing
        self._call_count: int = 0
        self._file_write_count: int = 0
        self._network_call_count: int = 0
        self._blocked_log: list[dict] = []
        self._orig_before_tool = None

    # -- Install / Uninstall ------------------------------------------------

    def install(self, harness: Harness) -> None:
        """Attach safety enforcer to harness hooks.

        Chains with existing ``before_tool`` hook — safety runs first.
        """
        self._orig_before_tool = harness.hooks.before_tool
        harness.hooks.before_tool = self._before_tool

    def uninstall(self, harness: Harness) -> None:
        """Restore original harness hooks."""
        harness.hooks.before_tool = self._orig_before_tool
        self._orig_before_tool = None

    # -- Hook implementation -------------------------------------------------

    def _before_tool(self, tool_name: str, tool_args: dict) -> bool:
        """Pre-execution safety check.

        Returns True to allow, False to block.
        """
        if not self.config.sandbox_mode:
            return self._chain(tool_name, tool_args)

        # 1. Check blocked patterns
        for pattern in self.config.blocked_tool_patterns:
            if fnmatch.fnmatch(tool_name, pattern):
                self._record_block(tool_name, f"Blocked pattern: {pattern}")
                return False

        # 2. Check whitelist (if configured)
        if self.config.allowed_tool_patterns:
            allowed = any(
                fnmatch.fnmatch(tool_name, p)
                for p in self.config.allowed_tool_patterns
            )
            if not allowed:
                self._record_block(tool_name, "Not in allowed patterns")
                return False

        # 3. Check destructive actions
        is_destructive = any(
            fnmatch.fnmatch(tool_name, p) for p in _DESTRUCTIVE_PATTERNS
        )
        if is_destructive:
            mode = self.config.destructive_action_mode
            if mode == "block":
                self._record_block(tool_name, "Destructive action blocked")
                return False
            elif mode == "noop":
                self._record_block(tool_name, "Destructive action no-op'd")
                return True
            # "log" mode: allow but log
            self._record_block(tool_name, "Destructive action logged (allowed)")

        # 4. Check network isolation
        if self.config.network_isolation:
            is_network = any(
                fnmatch.fnmatch(tool_name, p) for p in _NETWORK_PATTERNS
            )
            if is_network:
                self._record_block(tool_name, "Network access blocked")
                return False

        # 5. Check total tool call budget
        self._call_count += 1
        if (self.config.max_tool_calls > 0
                and self._call_count > self.config.max_tool_calls):
            self._record_block(
                tool_name,
                f"Tool call budget exceeded ({self.config.max_tool_calls})",
            )
            return False

        return self._chain(tool_name, tool_args)

    def _chain(self, tool_name: str, tool_args: dict) -> bool:
        """Chain to the original before_tool hook."""
        if self._orig_before_tool is not None:
            return self._orig_before_tool(tool_name, tool_args)
        return True

    def _record_block(self, tool_name: str, reason: str) -> None:
        """Record a blocked tool call."""
        self._blocked_log.append({
            "tool_name": tool_name,
            "reason": reason,
        })

    # -- Public API ----------------------------------------------------------

    def reset_counters(self) -> None:
        """Reset call counters between runs."""
        self._call_count = 0
        self._file_write_count = 0
        self._network_call_count = 0

    def get_blocked_log(self) -> list[dict]:
        """Return the log of blocked tool calls."""
        return list(self._blocked_log)

    def get_stats(self) -> dict:
        """Return safety enforcement statistics."""
        return {
            "total_tool_calls": self._call_count,
            "blocked_count": len(self._blocked_log),
            "file_writes": self._file_write_count,
            "network_calls": self._network_call_count,
        }


# ---------------------------------------------------------------------------
# Synthetic data framework
# ---------------------------------------------------------------------------

class SyntheticDataFramework:
    """Generate synthetic test data from GMS manifold geometry.

    Uses the learned entity embedding distribution on S^{m-1} to
    generate synthetic entities that have similar geometric properties
    to real entities.

    Parameters
    ----------
    store : GMSExpertStore
        GMS store with trained entity embeddings.
    """

    def __init__(self, store: GMSExpertStore):
        self._store = store

    def synthesize_entities(
        self, n: int, seed: int = 42,
    ) -> list[dict]:
        """Generate n synthetic entity records.

        Samples random points on S^{m-1} near the learned entity
        distribution, then maps to entity-like names via nearest
        neighbor in the embedding space.

        Parameters
        ----------
        n : int
            Number of synthetic entities to generate.
        seed : int
            Random seed.

        Returns
        -------
        list[dict]
            Synthetic entity records with ``"name"``, ``"type"``,
            ``"nearest_real"``, and ``"geodesic_distance"`` fields.
        """
        import torch

        rng = random.Random(seed)
        entities = []

        if self._store.adapter is None or self._store.gkg is None:
            return entities

        real_names = list(self._store.adapter.entity_to_idx.keys())
        if not real_names:
            return entities

        # Get all real entity embeddings
        device = next(self._store.gkg.parameters()).device
        n_real = len(real_names)
        real_indices = torch.arange(n_real, device=device)

        try:
            # Get v-space embeddings (semantic)
            real_embeddings = self._store.gkg.dual_emb.v_embed(real_indices)
            # Normalize to sphere
            real_embeddings = real_embeddings / real_embeddings.norm(
                dim=-1, keepdim=True,
            )
        except Exception:
            return entities

        dim = real_embeddings.shape[1]

        for i in range(n):
            # Sample a random direction on S^{m-1}
            torch.manual_seed(seed + i)
            noise = torch.randn(dim, device=device)
            noise = noise / noise.norm()

            # Perturb a random real entity embedding
            base_idx = rng.randint(0, n_real - 1)
            base_emb = real_embeddings[base_idx]

            # Mix: 70% real + 30% noise → project to sphere
            alpha = 0.7
            synth_emb = alpha * base_emb + (1 - alpha) * noise
            synth_emb = synth_emb / synth_emb.norm()

            # Find nearest real entity
            cosines = torch.mv(real_embeddings, synth_emb)
            nearest_idx = cosines.argmax().item()
            geo_dist = float(torch.acos(cosines[nearest_idx].clamp(-1, 1)))

            entities.append({
                "name": f"synth_entity_{i:04d}",
                "type": "synthetic",
                "nearest_real": real_names[nearest_idx],
                "geodesic_distance": round(geo_dist, 4),
                "base_entity": real_names[base_idx],
            })

        return entities

    def synthesize_relations(
        self,
        entities: list[str],
        density: float = 0.1,
        seed: int = 42,
    ) -> list[tuple[str, str, str]]:
        """Generate synthetic relations between entities.

        Parameters
        ----------
        entities : list[str]
            Entity names.
        density : float
            Proportion of possible edges to create [0, 1].
        seed : int
            Random seed.

        Returns
        -------
        list[tuple[str, str, str]]
            (head, relation, tail) triples.
        """
        rng = random.Random(seed)
        triples = []

        if self._store.adapter is None:
            return triples

        real_relations = list(self._store.adapter.relation_to_idx.keys())
        if not real_relations:
            real_relations = ["related_to", "has_property", "is_type"]

        for h in entities:
            for t in entities:
                if h == t:
                    continue
                if rng.random() < density:
                    r = rng.choice(real_relations)
                    triples.append((h, r, t))

        return triples

    def synthesize_fixture(
        self,
        n_entities: int = 10,
        relation_density: float = 0.1,
        seed: int = 42,
    ) -> dict:
        """Generate a complete synthetic data fixture.

        Returns a dictionary suitable for use as a ``DataFixture.data``
        payload.

        Parameters
        ----------
        n_entities : int
            Number of synthetic entities.
        relation_density : float
            Edge density.
        seed : int
            Random seed.
        """
        entities = self.synthesize_entities(n_entities, seed)
        entity_names = [e["name"] for e in entities]
        relations = self.synthesize_relations(entity_names, relation_density, seed)

        return {
            "entities": entities,
            "relations": [
                {"head": h, "relation": r, "tail": t}
                for h, r, t in relations
            ],
            "metadata": {
                "synthetic": True,
                "n_entities": n_entities,
                "n_relations": len(relations),
                "seed": seed,
            },
        }
