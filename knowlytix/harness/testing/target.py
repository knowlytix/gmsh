# SPDX-License-Identifier: Apache-2.0
"""GMSH Target Profile — formal SUT identity for black-box testing.

Defines the System Under Test (SUT) as a structured, versioned entity
so that campaigns, audit reports, and version drift detection can
reference it unambiguously.

A target profile captures:

- **Identity** — unique ID, human-readable name, semantic version
- **Type** — llm, agent, rag, or browser
- **Endpoint** — API endpoint or ``"local"`` for in-process adapters
- **Capabilities** — what the SUT can do (tool_use, multi_turn, etc.)
- **Isolation** — sandbox and resource constraints for safe testing
- **Timeout** — per-invocation and per-run limits

Profiles are loaded from YAML and passed to adapters, run records,
and audit reports.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import yaml


# ---------------------------------------------------------------------------
# Isolation requirements
# ---------------------------------------------------------------------------

@dataclass
class IsolationRequirements:
    """Resource and sandbox constraints for SUT execution.

    Attributes
    ----------
    network_isolated : bool
        If True, SUT may only access whitelisted test endpoints.
    sandbox_filesystem : bool
        If True, SUT file operations are redirected to a temp sandbox.
    synthetic_data_only : bool
        If True, only synthetic identities and test data are used.
    max_memory_mb : int
        Memory cap in MB (0 = unlimited).
    max_wall_time_s : int
        Wall-clock cap per run in seconds (0 = unlimited).
    """

    network_isolated: bool = False
    sandbox_filesystem: bool = False
    synthetic_data_only: bool = False
    max_memory_mb: int = 0
    max_wall_time_s: int = 0

    def to_dict(self) -> dict:
        return {
            "network_isolated": self.network_isolated,
            "sandbox_filesystem": self.sandbox_filesystem,
            "synthetic_data_only": self.synthetic_data_only,
            "max_memory_mb": self.max_memory_mb,
            "max_wall_time_s": self.max_wall_time_s,
        }

    @classmethod
    def from_dict(cls, d: dict) -> IsolationRequirements:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Target profile
# ---------------------------------------------------------------------------

VALID_TARGET_TYPES = {"llm", "agent", "rag", "browser"}


@dataclass
class TargetProfile:
    """Formal identity of the System Under Test (SUT).

    Attributes
    ----------
    target_id : str
        Unique identifier (auto-generated as ``tgt_<uuid8>`` if empty).
    name : str
        Human-readable label (e.g. "DS Agent v2.1").
    version : str
        Semantic version or commit hash of the SUT.
    target_type : str
        One of ``"llm"``, ``"agent"``, ``"rag"``, ``"browser"``.
    endpoint : str
        API endpoint URL or ``"local"`` for in-process execution.
    model_name : str
        Foundation model name (e.g. ``"claude-sonnet-4-20250514"``).
    capabilities : list[str]
        Declared SUT capabilities (``"tool_use"``, ``"multi_turn"``,
        ``"file_upload"``, ``"code_execution"``, etc.).
    timeout_ms : int
        Per-invocation timeout in milliseconds.
    max_concurrent : int
        Maximum parallel invocations allowed.
    isolation : IsolationRequirements
        Sandbox and resource constraints.
    metadata : dict
        Arbitrary key-value metadata.
    created_at : str
        ISO-ish creation timestamp.
    """

    target_id: str = ""
    name: str = ""
    version: str = ""
    target_type: str = "agent"
    endpoint: str = "local"
    model_name: str = ""
    capabilities: list[str] = field(default_factory=list)
    timeout_ms: int = 60_000
    max_concurrent: int = 1
    isolation: IsolationRequirements = field(
        default_factory=IsolationRequirements
    )
    metadata: dict = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        if not self.target_id:
            self.target_id = f"tgt_{uuid.uuid4().hex[:8]}"
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        if self.target_type not in VALID_TARGET_TYPES:
            raise ValueError(
                f"target_type must be one of {VALID_TARGET_TYPES}, "
                f"got {self.target_type!r}"
            )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "name": self.name,
            "version": self.version,
            "target_type": self.target_type,
            "endpoint": self.endpoint,
            "model_name": self.model_name,
            "capabilities": list(self.capabilities),
            "timeout_ms": self.timeout_ms,
            "max_concurrent": self.max_concurrent,
            "isolation": self.isolation.to_dict(),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TargetProfile:
        d = dict(d)
        iso_raw = d.pop("isolation", {})
        if isinstance(iso_raw, dict):
            iso = IsolationRequirements.from_dict(iso_raw)
        else:
            iso = IsolationRequirements()
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(isolation=iso, **filtered)

    @classmethod
    def from_yaml(cls, path: str) -> TargetProfile:
        """Load a target profile from a YAML file.

        The YAML may contain a top-level ``target`` key or be flat::

            target:
              name: "DS Agent v2.1"
              version: "0.4.0"
              target_type: agent
              model_name: claude-sonnet-4-20250514
              capabilities:
                - tool_use
                - multi_turn
              timeout_ms: 60000
              isolation:
                network_isolated: false
                synthetic_data_only: true
        """
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        if "target" in raw and isinstance(raw["target"], dict):
            raw = raw["target"]
        return cls.from_dict(raw)

    def to_yaml(self, path: str) -> None:
        """Save target profile to a YAML file."""
        with open(path, "w") as f:
            yaml.safe_dump(
                {"target": self.to_dict()}, f,
                default_flow_style=False, sort_keys=False,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def has_capability(self, cap: str) -> bool:
        """Check if the target declares a specific capability."""
        return cap in self.capabilities

    def summary(self) -> str:
        """One-line human-readable summary."""
        caps = ", ".join(self.capabilities) if self.capabilities else "none"
        return (
            f"{self.name} ({self.target_type}, v{self.version}) "
            f"— model={self.model_name}, caps=[{caps}]"
        )
