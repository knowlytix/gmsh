# SPDX-License-Identifier: Apache-2.0
"""GMSH Artifact Versioning — version tracking for test artifacts.

Provides a generic versioning layer for test artifacts (fault profiles,
policy rules, data fixtures) following the same pattern as ScenarioRegistry.

Each artifact is stored as a per-version YAML file with SHA-256 content
hashing for integrity.  The registry supports CRUD, version listing,
and cross-version diffing.

Usage::

    registry = ArtifactRegistry("artifacts/fault_profiles")
    artifact = VersionedArtifact(
        artifact_type="fault_profile",
        content={"name": "network_timeout", "delay_ms": 5000},
    )
    aid = registry.add(artifact)
    loaded = registry.get(aid)
    versions = registry.list_versions(aid)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field

import yaml


# ---------------------------------------------------------------------------
# Versioned artifact
# ---------------------------------------------------------------------------

@dataclass
class VersionedArtifact:
    """A versioned test artifact with content hashing.

    Attributes
    ----------
    artifact_id : str
        Unique identifier (auto-generated as ``art_<uuid8>`` if empty).
    artifact_type : str
        Type of artifact: ``"fault_profile"``, ``"policy_rule"``,
        ``"fixture"``, ``"replay_manifest"``, etc.
    version : int
        Monotonically increasing version number.
    content : dict
        The artifact payload.
    schema : dict
        Optional JSON schema for validation.
    content_hash : str
        SHA-256 hash of the serialized content (auto-computed).
    created_at : str
        ISO timestamp of creation.
    updated_at : str
        ISO timestamp of last update.
    author : str
        Who created/updated this artifact.
    metadata : dict
        Arbitrary metadata.
    """

    artifact_id: str = ""
    artifact_type: str = ""
    version: int = 1
    content: dict = field(default_factory=dict)
    schema: dict = field(default_factory=dict)
    content_hash: str = ""
    created_at: str = ""
    updated_at: str = ""
    author: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.artifact_id:
            self.artifact_id = f"art_{uuid.uuid4().hex[:8]}"
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        if not self.content_hash and self.content:
            self.content_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of serialized content."""
        serialized = json.dumps(self.content, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "version": self.version,
            "content": self.content,
            "schema": self.schema,
            "content_hash": self.content_hash,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "author": self.author,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> VersionedArtifact:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_yaml(self) -> str:
        return yaml.dump(
            self.to_dict(),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    @classmethod
    def from_yaml_file(cls, path: str) -> VersionedArtifact:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def verify_integrity(self) -> bool:
        """Verify content hash matches current content."""
        return self.content_hash == self._compute_hash()


# ---------------------------------------------------------------------------
# Artifact registry
# ---------------------------------------------------------------------------

class ArtifactRegistry:
    """Manages versioned artifacts stored as YAML files.

    Mirrors the ScenarioRegistry pattern: each artifact is stored as
    ``{artifact_id}_v{version}.yaml`` with an ``index.json`` for fast
    lookups.

    Parameters
    ----------
    artifacts_dir : str
        Directory where YAML artifact files are stored.
    artifact_type : str
        Default artifact type for this registry instance.
    """

    INDEX_FILE = "index.json"

    def __init__(self, artifacts_dir: str, artifact_type: str = ""):
        self.artifacts_dir = artifacts_dir
        self.artifact_type = artifact_type
        self._index: dict = {}

        os.makedirs(artifacts_dir, exist_ok=True)
        self._load_index()

    # -- Index persistence -------------------------------------------------

    def _index_path(self) -> str:
        return os.path.join(self.artifacts_dir, self.INDEX_FILE)

    def _load_index(self):
        path = self._index_path()
        if os.path.isfile(path):
            with open(path) as f:
                self._index = json.load(f)
        else:
            self._index = {}

    def _save_index(self):
        path = self._index_path()
        data = {
            "artifacts": self._index,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # -- File naming -------------------------------------------------------

    def _yaml_filename(self, artifact_id: str, version: int) -> str:
        return f"{artifact_id}_v{version}.yaml"

    def _yaml_path(self, artifact_id: str, version: int) -> str:
        return os.path.join(
            self.artifacts_dir,
            self._yaml_filename(artifact_id, version),
        )

    # -- CRUD --------------------------------------------------------------

    def add(self, artifact: VersionedArtifact) -> str:
        """Save a new artifact and update the index.

        Parameters
        ----------
        artifact : VersionedArtifact
            The artifact to add.

        Returns
        -------
        str
            The artifact_id.
        """
        if self.artifact_type and not artifact.artifact_type:
            artifact.artifact_type = self.artifact_type

        artifact.content_hash = artifact._compute_hash()
        artifact.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S")

        aid = artifact.artifact_id
        path = self._yaml_path(aid, artifact.version)
        with open(path, "w") as f:
            f.write(artifact.to_yaml())

        self._index[aid] = {
            "latest_version": artifact.version,
            "artifact_type": artifact.artifact_type,
            "content_hash": artifact.content_hash,
            "updated_at": artifact.updated_at,
            "author": artifact.author,
        }
        self._save_index()
        return aid

    def get(self, artifact_id: str) -> VersionedArtifact | None:
        """Load the latest version of an artifact."""
        entry = self._index.get(artifact_id)
        if entry is None:
            return None
        version = entry["latest_version"]
        path = self._yaml_path(artifact_id, version)
        if not os.path.isfile(path):
            return None
        return VersionedArtifact.from_yaml_file(path)

    def get_version(
        self, artifact_id: str, version: int,
    ) -> VersionedArtifact | None:
        """Load a specific version of an artifact."""
        path = self._yaml_path(artifact_id, version)
        if not os.path.isfile(path):
            return None
        return VersionedArtifact.from_yaml_file(path)

    def update(self, artifact: VersionedArtifact) -> int:
        """Increment version and save a new YAML file.

        Returns
        -------
        int
            The new version number.
        """
        entry = self._index.get(artifact.artifact_id)
        if entry is not None:
            artifact.version = entry["latest_version"] + 1
        else:
            artifact.version = 1

        artifact.content_hash = artifact._compute_hash()
        artifact.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S")

        path = self._yaml_path(artifact.artifact_id, artifact.version)
        with open(path, "w") as f:
            f.write(artifact.to_yaml())

        self._index[artifact.artifact_id] = {
            "latest_version": artifact.version,
            "artifact_type": artifact.artifact_type,
            "content_hash": artifact.content_hash,
            "updated_at": artifact.updated_at,
            "author": artifact.author,
        }
        self._save_index()
        return artifact.version

    def delete(self, artifact_id: str):
        """Remove from index (YAML files kept for history)."""
        if artifact_id in self._index:
            del self._index[artifact_id]
            self._save_index()

    def list_all(self) -> list[VersionedArtifact]:
        """Return all artifacts at their latest version."""
        artifacts = []
        for aid in self._index:
            art = self.get(aid)
            if art is not None:
                artifacts.append(art)
        return artifacts

    def list_versions(self, artifact_id: str) -> list[int]:
        """List all available version numbers for an artifact."""
        versions = []
        prefix = f"{artifact_id}_v"
        for fname in os.listdir(self.artifacts_dir):
            if fname.startswith(prefix) and fname.endswith(".yaml"):
                ver_str = fname[len(prefix):-len(".yaml")]
                try:
                    versions.append(int(ver_str))
                except ValueError:
                    continue
        return sorted(versions)

    def diff(
        self, artifact_id: str, version_a: int, version_b: int,
    ) -> dict:
        """Compare two versions of an artifact.

        Returns a dict with ``added``, ``removed``, and ``changed``
        keys describing the content differences.
        """
        a = self.get_version(artifact_id, version_a)
        b = self.get_version(artifact_id, version_b)

        if a is None or b is None:
            return {"error": f"Version not found: a={a is not None}, b={b is not None}"}

        added = {}
        removed = {}
        changed = {}

        all_keys = set(a.content.keys()) | set(b.content.keys())
        for key in all_keys:
            in_a = key in a.content
            in_b = key in b.content
            if in_a and not in_b:
                removed[key] = a.content[key]
            elif in_b and not in_a:
                added[key] = b.content[key]
            elif a.content[key] != b.content[key]:
                changed[key] = {
                    "from": a.content[key],
                    "to": b.content[key],
                }

        return {
            "artifact_id": artifact_id,
            "version_a": version_a,
            "version_b": version_b,
            "hash_a": a.content_hash,
            "hash_b": b.content_hash,
            "added": added,
            "removed": removed,
            "changed": changed,
        }


# ---------------------------------------------------------------------------
# Typed registry subclasses
# ---------------------------------------------------------------------------

class FaultProfileRegistry(ArtifactRegistry):
    """Registry for fault profile artifacts."""

    def __init__(self, artifacts_dir: str):
        super().__init__(artifacts_dir, artifact_type="fault_profile")


class PolicyRuleRegistry(ArtifactRegistry):
    """Registry for policy rule artifacts."""

    def __init__(self, artifacts_dir: str):
        super().__init__(artifacts_dir, artifact_type="policy_rule")


class FixtureRegistry(ArtifactRegistry):
    """Registry for data fixture artifacts."""

    def __init__(self, artifacts_dir: str):
        super().__init__(artifacts_dir, artifact_type="fixture")
