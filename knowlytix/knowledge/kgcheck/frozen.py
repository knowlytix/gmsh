# SPDX-License-Identifier: Apache-2.0
"""Recursive freezing of JSON-like data, so a snapshot cannot change under a rule.

``freeze`` copies: objects become read-only mappings and arrays become tuples,
so neither later changes to the caller's input nor writes through the result
reach the frozen value. ``thaw`` gives a plain, mutable copy for serialization.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

FrozenJson = str | int | float | bool | None | tuple["FrozenJson", ...] | Mapping[str, "FrozenJson"]


def freeze(value: object) -> FrozenJson:
    """A deep, read-only copy of a JSON-like value."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    return value  # type: ignore[return-value]  # scalars are immutable


def freeze_mapping(value: Mapping[str, object]) -> Mapping[str, FrozenJson]:
    """``freeze`` for a mapping, typed as one."""
    frozen = freeze(value)
    assert isinstance(frozen, Mapping)
    return frozen


def thaw(value: FrozenJson) -> object:
    """A plain dict/list copy of a frozen value."""
    if isinstance(value, Mapping):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [thaw(v) for v in value]
    return value
