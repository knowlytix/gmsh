# SPDX-License-Identifier: Apache-2.0
"""Raw artifact reading that reports problems instead of hiding them.

Duplicate JSON keys and non-finite numbers are accepted by Python's ``json``
module without a trace; here each becomes a diagnostic. Only the named file
inside the store directory is read: a path that resolves outside it is refused.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from knowlytix.knowledge.kgcheck.models import (
    ArtifactRecord,
    Diagnostic,
    JsonValue,
    RawRef,
)


@dataclass(frozen=True)
class RawArtifact:
    """One artifact as read. ``value`` is set only when the file parsed."""

    record: ArtifactRecord
    value: JsonValue = None
    parsed: bool = False
    text: str | None = None
    diagnostics: tuple[Diagnostic, ...] = ()


def pointer_token(key: str | int) -> str:
    """RFC 6901 escaping of one JSON pointer segment."""
    return str(key).replace("~", "~0").replace("/", "~1")


def _safe_path(directory: Path, name: str) -> Path | None:
    path = directory / name
    root = directory.resolve()
    return path if path.resolve().is_relative_to(root) else None


def read_bytes(directory: Path, name: str, required: bool) -> tuple[ArtifactRecord, bytes | None,
                                                                    list[Diagnostic]]:
    """Read one file's bytes, with a record of presence, size and digest."""
    try:
        path = _safe_path(directory, name)
    except (RuntimeError, OSError) as exc:  # e.g. a symlink loop
        return (ArtifactRecord(name, True, required), None,
                [Diagnostic("unreadable_artifact", f"{type(exc).__name__}: {exc}", RawRef(name))])
    if path is None:
        return (ArtifactRecord(name, False, required),
                None, [Diagnostic("outside_store", "path resolves outside the store", RawRef(name))])
    if path.exists() and not path.is_file():
        return (ArtifactRecord(name, True, required), None,
                [Diagnostic("unreadable_artifact", "not a regular file", RawRef(name))])
    if not path.is_file():
        diags = [Diagnostic("missing_required_artifact", "required file is absent", RawRef(name))]
        return ArtifactRecord(name, False, required), None, diags if required else []
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return (ArtifactRecord(name, False, required), None,
                [Diagnostic("disappeared_during_read", "file vanished before it was read", RawRef(name))])
    except OSError as exc:
        return (ArtifactRecord(name, True, required), None,
                [Diagnostic("unreadable_artifact", f"{type(exc).__name__}: {exc}", RawRef(name))])
    record = ArtifactRecord(name, True, required, len(data), hashlib.sha256(data).hexdigest())
    return record, data, []


def _decode(data: bytes, name: str) -> tuple[str | None, list[Diagnostic]]:
    try:
        return data.decode("utf-8"), []
    except UnicodeDecodeError as exc:
        return None, [Diagnostic("invalid_utf8", str(exc), RawRef(name))]


class _Repeated(dict):  # type: ignore[type-arg]
    """An object whose JSON text repeated keys; ``repeated`` lists them."""

    repeated: tuple[str, ...] = ()


def parse_json(text: str, name: str) -> tuple[JsonValue, bool, list[Diagnostic]]:
    """Parse, recording duplicate keys (first value kept), non-finite or oversized
    numbers and lone surrogate escapes, each at its JSON pointer where one exists."""
    diags: list[Diagnostic] = []

    def pairs(items: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
        out: dict[str, JsonValue] = {}
        repeated = []
        for key, value in items:
            if key in out:
                repeated.append(key)
                continue
            out[key] = value
        if not repeated:
            return out
        marked = _Repeated(out)
        marked.repeated = tuple(repeated)
        return marked

    def integer(token: str) -> int | float:
        if len(token.lstrip("-")) > _MAX_INT_DIGITS:
            diags.append(Diagnostic("number_out_of_range", f"an integer of {len(token)} digits "
                                    "exceeds what the reader converts", RawRef(name)))
            return math.inf
        return int(token)

    def constant(token: str) -> float:
        diags.append(Diagnostic("nonfinite_number", f"{token} is not valid JSON", RawRef(name)))
        return float(token.replace("Infinity", "inf"))

    def real(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            diags.append(Diagnostic("number_out_of_range", f"{token[:40]} overflows a double",
                                    RawRef(name)))
        return value

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant,
                           parse_float=real, parse_int=integer)
    except (ValueError, RecursionError) as exc:  # JSONDecodeError is a ValueError
        return None, False, [*diags, Diagnostic("invalid_json", f"{type(exc).__name__}: {exc}"[:300],
                                                RawRef(name))]
    return value, True, diags + _located(value, name)


_MAX_INT_DIGITS = 4000  # below CPython's int-string limit (4300)


def _located(value: JsonValue, name: str) -> list[Diagnostic]:
    """Diagnostics that need a pointer: repeated keys and lone surrogates."""
    out: list[Diagnostic] = []
    for pointer, item in _walk(value, ""):
        if isinstance(item, _Repeated):
            out += [Diagnostic("duplicate_key", f"key {k!r} repeated; the first value is kept",
                               RawRef(name, f"{pointer}/{pointer_token(k)}")) for k in item.repeated]
        texts = [*item.keys(), *(v for v in item.values() if isinstance(v, str))] \
            if isinstance(item, dict) else [item] if isinstance(item, str) else []
        if any(_lone_surrogate(t) for t in texts):
            out.append(Diagnostic("lone_surrogate", "a string holds an unpaired UTF-16 surrogate "
                                  "escape, which is not valid Unicode text", RawRef(name, pointer)))
    return out


def _walk(value: JsonValue, pointer: str) -> Iterator[tuple[str, JsonValue]]:
    """Every container and string with its pointer, iteratively (no recursion limit)."""
    stack = [(pointer, value)]
    while stack:
        pointer, item = stack.pop()
        yield pointer, item
        if isinstance(item, dict):
            stack += [(f"{pointer}/{pointer_token(k)}", v) for k, v in item.items()
                      if isinstance(v, (dict, list))]
        elif isinstance(item, list):
            stack += [(f"{pointer}/{i}", v) for i, v in enumerate(item) if isinstance(v, (dict, list, str))]


def _lone_surrogate(text: str) -> bool:
    return any(0xD800 <= ord(c) <= 0xDFFF for c in text)


def read_json_artifact(directory: Path, name: str, required: bool) -> RawArtifact:
    record, data, diags = read_bytes(directory, name, required)
    if data is None:
        return RawArtifact(record, diagnostics=tuple(diags))
    text, more = _decode(data, name)
    if text is None:
        return RawArtifact(record, diagnostics=tuple(diags + more))
    value, ok, more = parse_json(text, name)
    return RawArtifact(record, value, ok, text, tuple(diags + more))


def json_artifact_from_text(name: str, text: str, required: bool = True) -> RawArtifact:
    """An artifact parsed from text held in memory, recorded as if read from ``name``."""
    data = text.encode("utf-8", "surrogatepass")
    record = ArtifactRecord(name, True, required, len(data), hashlib.sha256(data).hexdigest())
    value, ok, diags = parse_json(text, name)
    return RawArtifact(record, value, ok, text, tuple(diags))


def read_text_artifact(directory: Path, name: str) -> RawArtifact:
    record, data, diags = read_bytes(directory, name, False)
    if data is None:
        return RawArtifact(record, diagnostics=tuple(diags))
    text, more = _decode(data, name)
    return RawArtifact(record, text=text, diagnostics=tuple(diags + more))


def lexical_numbers(text: str) -> JsonValue:
    """The document with every number left as its exact source text (first key wins)."""
    def first(items: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
        out: dict[str, JsonValue] = {}
        for key, value in items:
            out.setdefault(key, value)
        return out
    try:
        return json.loads(text, object_pairs_hook=first, parse_float=str, parse_int=str,
                          parse_constant=str)
    except (ValueError, RecursionError):
        return None


def finite_float(value: JsonValue) -> float | None:
    """A JSON number as a finite float, or None (booleans, overflow and non-finite included)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        converted = float(value)
    except OverflowError:
        return None
    return converted if math.isfinite(converted) else None
