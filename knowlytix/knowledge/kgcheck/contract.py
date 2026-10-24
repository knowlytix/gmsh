# SPDX-License-Identifier: Apache-2.0
"""The OntoDisc store artifact contract, as shipped JSON Schema.

``schemas/ontodisc_artifacts.schema.json`` holds one ``$defs`` entry per persisted
JSON file and an ``x-artifacts`` index from file name to definition. The schema
fixes structure only; semantic problems are audit findings. Validating against it
needs a JSON Schema (draft 2020-12) implementation, which this package does not
import: the store loader checks rows itself so it can report each malformed row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib.resources import files

CONTRACT_FILE = "ontodisc_artifacts.schema.json"

JsonObject = dict[str, object]


@dataclass(frozen=True)
class ArtifactSpec:
    """One persisted file: its name, schema definition and whether a store needs it."""

    filename: str
    definition: str
    required: bool


@cache
def _contract_text() -> str:
    return files("knowlytix.knowledge.kgcheck").joinpath("schemas", CONTRACT_FILE).read_text(
        encoding="utf-8")


def artifact_contract() -> JsonObject:
    """The full contract document (a fresh copy on every call)."""
    return json.loads(_contract_text())


def contract_version() -> str:
    """The contract version, e.g. ``"ontodisc-artifacts/1"``."""
    return str(artifact_contract()["x-contract-version"])


def artifact_specs() -> tuple[ArtifactSpec, ...]:
    """Every persisted JSON file the contract covers, in contract order."""
    index = artifact_contract()["x-artifacts"]
    assert isinstance(index, dict)
    return tuple(ArtifactSpec(name, str(entry["def"]), bool(entry["required"]))
                 for name, entry in index.items())


def artifact_schema(filename: str) -> JsonObject:
    """A standalone schema for one artifact file, resolving within the contract.

    Raises:
        KeyError: ``filename`` is not an artifact the contract covers.
    """
    contract = artifact_contract()
    spec = {s.filename: s for s in artifact_specs()}[filename]
    return {"$schema": contract["$schema"], "$defs": contract["$defs"],
            "$ref": f"#/$defs/{spec.definition}"}
