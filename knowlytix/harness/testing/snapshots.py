# SPDX-License-Identifier: Apache-2.0
"""GMSH Snapshot Manager — versioned GMS store snapshots with diff.

Captures a complete copy of a GMS store at a point in time (ingest, pre-run)
so that benchmark results can be reproduced and store evolution tracked.

Each snapshot records:
- Full store directory copy (model.pt, adapter.json, triples.json, enm.json, etc.)
- Metadata: run_id, timestamp, store stats, parent snapshot
- GMS triples in the tracing store (snapshot_id, "is_type", "gms_snapshot")

Snapshots are stored under ``snapshot_dir/<snapshot_id>/`` and registered
in a ``manifest.json`` index file.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore

    from .replay import ReplayManifest
    from .tracing import DOETracingGMS


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GMSSnapshot:
    """Record of a GMS store snapshot."""

    snapshot_id: str = ""
    run_id: str = ""
    timestamp: str = ""
    store_path: str = ""          # original store path
    snapshot_path: str = ""       # path to the snapshot copy
    parent_snapshot_id: str = ""
    trigger: str = ""             # "ingest", "pre_run", "manual"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.snapshot_id:
            self.snapshot_id = f"snap_{uuid.uuid4().hex[:12]}"
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "store_path": self.store_path,
            "snapshot_path": self.snapshot_path,
            "parent_snapshot_id": self.parent_snapshot_id,
            "trigger": self.trigger,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GMSSnapshot":
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Snapshot Manager
# ---------------------------------------------------------------------------

class SnapshotManager:
    """Manages versioned snapshots of GMS stores.

    Parameters
    ----------
    snapshot_dir : str
        Root directory for all snapshots.
    tracing : DOETracingGMS or None
        If provided, snapshot events are recorded as GMS triples.
    """

    MANIFEST_FILE = "manifest.json"

    def __init__(
        self,
        snapshot_dir: str,
        tracing: "DOETracingGMS | None" = None,
    ):
        self.snapshot_dir = snapshot_dir
        self.tracing = tracing
        self._snapshots: list[GMSSnapshot] = []

        os.makedirs(snapshot_dir, exist_ok=True)
        self._load_manifest()

    # ----- Public API -----------------------------------------------------

    def take_snapshot(
        self,
        store: "GMSExpertStore",
        run_id: str = "",
        trigger: str = "manual",
    ) -> GMSSnapshot:
        """Copy the current GMS store to a timestamped snapshot directory.

        Parameters
        ----------
        store : GMSExpertStore
            The store to snapshot.
        run_id : str
            Benchmark run identifier (for lineage).
        trigger : str
            What triggered the snapshot ("ingest", "pre_run", "manual").

        Returns
        -------
        GMSSnapshot
            The created snapshot record.
        """
        # Ensure store is saved before copying
        store.save()

        parent_id = self._snapshots[-1].snapshot_id if self._snapshots else ""

        snap = GMSSnapshot(
            run_id=run_id,
            store_path=store.store_path,
            parent_snapshot_id=parent_id,
            trigger=trigger,
            metadata=store.stats(),
        )

        # Copy store directory
        snap.snapshot_path = os.path.join(self.snapshot_dir, snap.snapshot_id)
        shutil.copytree(store.store_path, snap.snapshot_path)

        # Compute content hash for integrity
        snap.metadata["content_hash"] = self._hash_directory(snap.snapshot_path)

        self._snapshots.append(snap)
        self._save_manifest()

        # Trace snapshot event via public API
        if self.tracing:
            self.tracing.trace_snapshot(
                snapshot_id=snap.snapshot_id,
                run_id=snap.run_id,
                store_stats=snap.metadata,
            )

        print(f"  Snapshot {snap.snapshot_id} ({trigger}): "
              f"{snap.metadata.get('entities', '?')} entities, "
              f"{snap.metadata.get('triples', '?')} triples")

        return snap

    def restore_snapshot(
        self,
        snapshot_id: str,
        device=None,
    ) -> "GMSExpertStore":
        """Load a GMS store from a snapshot.

        Parameters
        ----------
        snapshot_id : str
            The snapshot to restore.
        device : torch.device or None
            Device for model loading. Defaults to CPU.

        Returns
        -------
        GMSExpertStore
            A fully loaded store from the snapshot.
        """
        snap = self._get_snapshot(snapshot_id)
        if snap is None:
            raise ValueError(f"Snapshot not found: {snapshot_id}")

        if not os.path.isdir(snap.snapshot_path):
            raise FileNotFoundError(
                f"Snapshot directory missing: {snap.snapshot_path}")

        import torch

        from knowlytix.knowledge.config import DocGMSConfig
        from knowlytix.knowledge.store import GMSExpertStore

        if device is None:
            device = torch.device("cpu")

        config = DocGMSConfig(store_path=snap.snapshot_path)
        store = GMSExpertStore(config, device)
        store.load()

        return store

    def list_snapshots(self) -> list[GMSSnapshot]:
        """Return all snapshots, oldest first."""
        return list(self._snapshots)

    def diff_snapshots(
        self,
        id_a: str,
        id_b: str,
    ) -> dict:
        """Compare two snapshots and return a diff summary.

        Returns
        -------
        dict
            Keys: entities_added, entities_removed, triples_added,
            triples_removed, enm_added, enm_removed, stats_a, stats_b.
        """
        snap_a = self._get_snapshot(id_a)
        snap_b = self._get_snapshot(id_b)
        if snap_a is None:
            raise ValueError(f"Snapshot not found: {id_a}")
        if snap_b is None:
            raise ValueError(f"Snapshot not found: {id_b}")

        entities_a = self._load_entities(snap_a.snapshot_path)
        entities_b = self._load_entities(snap_b.snapshot_path)
        triples_a = self._load_triples(snap_a.snapshot_path)
        triples_b = self._load_triples(snap_b.snapshot_path)
        enm_a = self._load_enm_keys(snap_a.snapshot_path)
        enm_b = self._load_enm_keys(snap_b.snapshot_path)

        return {
            "snapshot_a": id_a,
            "snapshot_b": id_b,
            "entities_added": sorted(entities_b - entities_a),
            "entities_removed": sorted(entities_a - entities_b),
            "triples_added": sorted(triples_b - triples_a),
            "triples_removed": sorted(triples_a - triples_b),
            "enm_added": sorted(enm_b - enm_a),
            "enm_removed": sorted(enm_a - enm_b),
            "stats_a": snap_a.metadata,
            "stats_b": snap_b.metadata,
        }

    def get_latest(self) -> GMSSnapshot | None:
        """Return the most recent snapshot, or None."""
        return self._snapshots[-1] if self._snapshots else None

    def create_replay_manifest(
        self,
        snapshot_id: str,
        run_id: str = "",
        config_snapshot: dict | None = None,
        doe_questions_path: str = "",
        fault_profiles: list[dict] | None = None,
        policy_rules: list[dict] | None = None,
        reference_result_path: str = "",
        random_seed: int = 42,
    ) -> "ReplayManifest":
        """Create a replay manifest from a snapshot and run metadata.

        The manifest captures all inputs needed to deterministically
        re-execute a benchmark run: GMS store snapshot, config, question
        list, and execution context (faults, policies, seed).

        Parameters
        ----------
        snapshot_id : str
            Snapshot to replay from.
        run_id : str
            Source run ID (for provenance).
        config_snapshot : dict or None
            Frozen config at the time of the run.
        doe_questions_path : str
            Path to serialised question list.
        fault_profiles : list[dict] or None
            Fault profile dicts to apply during replay.
        policy_rules : list[dict] or None
            Policy rule dicts to enforce during replay.
        reference_result_path : str
            Path to the reference result for comparison.
        random_seed : int
            Seed for deterministic execution.

        Returns
        -------
        ReplayManifest
            The created manifest.
        """
        snap = self._get_snapshot(snapshot_id)
        if snap is None:
            raise ValueError(f"Snapshot not found: {snapshot_id}")

        from .replay import ReplayManifest

        manifest = ReplayManifest(
            source_run_id=run_id,
            config_snapshot=config_snapshot or {},
            snapshot_id=snapshot_id,
            doe_questions_path=doe_questions_path,
            fault_profiles=fault_profiles or [],
            policy_rules=policy_rules or [],
            random_seed=random_seed,
            reference_result_path=reference_result_path,
        )

        # Trace manifest creation
        if self.tracing:
            self.tracing.trace_regression_promotion(
                run_id=run_id or "unknown",
                scenario_id=manifest.manifest_id,
                source_question_id="replay_manifest",
                failure_code=f"manifest_from_{snapshot_id}",
            )

        return manifest

    # ----- Internal helpers -----------------------------------------------

    def _get_snapshot(self, snapshot_id: str) -> GMSSnapshot | None:
        for snap in self._snapshots:
            if snap.snapshot_id == snapshot_id:
                return snap
        return None

    def _load_manifest(self):
        manifest_path = os.path.join(self.snapshot_dir, self.MANIFEST_FILE)
        if os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                data = json.load(f)
            self._snapshots = [
                GMSSnapshot.from_dict(s) for s in data.get("snapshots", [])
            ]

    def _save_manifest(self):
        manifest_path = os.path.join(self.snapshot_dir, self.MANIFEST_FILE)
        data = {
            "snapshots": [s.to_dict() for s in self._snapshots],
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(manifest_path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _hash_directory(dir_path: str) -> str:
        """Compute a SHA-256 hash over all files in a directory."""
        h = hashlib.sha256()
        for root, _dirs, files in sorted(os.walk(dir_path)):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            h.update(chunk)
                except (OSError, PermissionError):
                    continue
        return h.hexdigest()

    @staticmethod
    def _load_entities(snapshot_path: str) -> set[str]:
        """Load entity set from a snapshot's adapter.json."""
        adapter_path = os.path.join(snapshot_path, "adapter.json")
        if not os.path.isfile(adapter_path):
            return set()
        with open(adapter_path) as f:
            data = json.load(f)
        return set(data.get("entity_to_idx", {}).keys())

    @staticmethod
    def _load_triples(snapshot_path: str) -> set[tuple[str, str, str]]:
        """Load triple set from a snapshot's triples.json."""
        triples_path = os.path.join(snapshot_path, "triples.json")
        if not os.path.isfile(triples_path):
            return set()
        with open(triples_path) as f:
            data = json.load(f)
        return {tuple(t) for t in data if len(t) == 3}

    @staticmethod
    def _load_enm_keys(snapshot_path: str) -> set[str]:
        """Load ENM key set from a snapshot's enm.json."""
        enm_path = os.path.join(snapshot_path, "enm.json")
        if not os.path.isfile(enm_path):
            return set()
        with open(enm_path) as f:
            data = json.load(f)
        return set(data.keys())
