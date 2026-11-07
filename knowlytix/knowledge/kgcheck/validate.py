# SPDX-License-Identifier: Apache-2.0
"""Validation of parsed artifacts against the shipped contract, without dependencies.

Implements exactly the JSON Schema (draft 2020-12) keywords the artifact
contract uses, so an audit does not need a schema library. A keyword outside
that set raises ``UnsupportedKeyword`` instead of being ignored, so a contract
change cannot silently weaken validation. Only local ``#/$defs/...``
references are resolved. Every violation carries the JSON pointer of the
offending value.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import cache

from knowlytix.knowledge.kgcheck.adapters._json import pointer_token

Schema = Mapping[str, object]

ANNOTATIONS = frozenset({"description", "title", "$schema", "$id", "$defs", "$comment"})


class UnsupportedKeyword(ValueError):
    """The schema uses a keyword this validator does not implement."""


@dataclass(frozen=True)
class Violation:
    pointer: str
    message: str


def _is_type(value: object, name: str) -> bool:
    if name == "null":
        return value is None
    if name == "boolean":
        return isinstance(value, bool)
    if name == "string":
        return isinstance(value, str)
    if name == "array":
        return isinstance(value, (list, tuple))
    if name == "object":
        return isinstance(value, Mapping)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if name == "number" or isinstance(value, int):
        return name in ("number", "integer")
    return name == "integer" and value.is_integer()  # a float such as 1.0 is an integer


class Validator:
    """Validates values against one schema document (its ``$defs`` resolve locally)."""

    def __init__(self, document: Schema) -> None:
        self._defs = document.get("$defs", {})
        self._keywords: dict[str, Callable[[object, Schema, str], Iterator[Violation]]] = {
            "type": self._type, "enum": self._enum, "const": self._const, "$ref": self._ref,
            "properties": self._properties, "required": self._required,
            "additionalProperties": self._additional, "propertyNames": self._names,
            "items": self._items, "minItems": self._min_items, "maxItems": self._max_items,
            "minLength": self._min_length, "pattern": self._pattern, "minimum": self._minimum,
            "maximum": self._maximum,
            "oneOf": self._one_of, "if": self._if, "then": _nothing,
        }

    def errors(self, value: object, schema: Schema, pointer: str = "") -> list[Violation]:
        return list(self._validate(value, schema, pointer))

    def _validate(self, value: object, schema: Schema, pointer: str) -> Iterator[Violation]:
        for key in schema:
            if key in ANNOTATIONS:
                continue
            check = self._keywords.get(key)
            if check is None:
                raise UnsupportedKeyword(f"schema keyword {key!r} is not implemented")
            yield from check(value, schema, pointer)

    def _ref(self, value: object, schema: Schema, pointer: str) -> Iterator[Violation]:
        ref = str(schema["$ref"])
        if not ref.startswith("#/$defs/"):
            raise UnsupportedKeyword(f"only local $defs references are supported, not {ref!r}")
        yield from self._validate(value, self._defs[ref.removeprefix("#/$defs/")], pointer)

    def _type(self, value: object, schema: Schema, pointer: str) -> Iterator[Violation]:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is_type(value, str(n)) for n in names):
            yield Violation(pointer, f"expected {' or '.join(map(str, names))}, got {_kind(value)}")

    def _enum(self, value: object, schema: Schema, pointer: str) -> Iterator[Violation]:
        options = schema["enum"]
        if not any(_equal(value, o) for o in options):  # type: ignore[union-attr]
            yield Violation(pointer, f"{_short(value)} is not one of {options}")

    def _const(self, value: object, schema: Schema, pointer: str) -> Iterator[Violation]:
        if not _equal(value, schema["const"]):
            yield Violation(pointer, f"{_short(value)} must be {schema['const']!r}")

    def _properties(self, value: object, schema: Schema, pointer: str) -> Iterator[Violation]:
        if isinstance(value, Mapping):
            for key, sub in schema["properties"].items():  # type: ignore[union-attr]
                if key in value:
                    yield from self._validate(value[key], sub, f"{pointer}/{pointer_token(key)}")

    def _required(self, value: object, schema: Schema, pointer: str) -> Iterator[Violation]:
        if isinstance(value, Mapping):
            for key in schema["required"]:  # type: ignore[union-attr]
                if key not in value:
                    yield Violation(pointer, f"required field {key!r} is missing")

    def _additional(self, value: object, schema: Schema, pointer: str) -> Iterator[Violation]:
        if not isinstance(value, Mapping):
            return
        rule, known = schema["additionalProperties"], schema.get("properties", {})
        for key, item in value.items():
            if key in known:  # type: ignore[operator]
                continue
            if rule is False:
                yield Violation(f"{pointer}/{pointer_token(key)}", f"unexpected field {key!r}")
            elif isinstance(rule, Mapping):
                yield from self._validate(item, rule, f"{pointer}/{pointer_token(key)}")

    def _names(self, value: object, schema: Schema, pointer: str) -> Iterator[Violation]:
        if isinstance(value, Mapping):
            for key in value:
                for v in self._validate(key, schema["propertyNames"], pointer):  # type: ignore[arg-type]
                    yield Violation(f"{pointer}/{pointer_token(key)}", f"key {key!r}: {v.message}")

    def _items(self, value: object, schema: Schema, pointer: str) -> Iterator[Violation]:
        if isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                yield from self._validate(item, schema["items"], f"{pointer}/{i}")  # type: ignore[arg-type]

    def _min_items(self, value: object, schema: Schema, pointer: str) -> Iterator[Violation]:
        if isinstance(value, (list, tuple)) and len(value) < schema["minItems"]:  # type: ignore[operator]
            yield Violation(pointer, f"needs at least {schema['minItems']} items, has {len(value)}")

    def _max_items(self, value: object, schema: Schema, pointer: str) -> Iterator[Violation]:
        if isinstance(value, (list, tuple)) and len(value) > schema["maxItems"]:  # type: ignore[operator]
            yield Violation(pointer, f"allows at most {schema['maxItems']} items, has {len(value)}")

    def _min_length(self, value: object, schema: Schema, pointer: str) -> Iterator[Violation]:
        if isinstance(value, str) and len(value) < schema["minLength"]:  # type: ignore[operator]
            yield Violation(pointer, f"must have at least {schema['minLength']} characters")

    def _pattern(self, value: object, schema: Schema, pointer: str) -> Iterator[Violation]:
        if isinstance(value, str) and not _ecma_pattern(str(schema["pattern"])).search(value):
            yield Violation(pointer, f"{_short(value)} does not match {schema['pattern']!r}")

    def _minimum(self, value: object, schema: Schema, pointer: str) -> Iterator[Violation]:
        if _is_type(value, "number") and value < schema["minimum"]:  # type: ignore[operator]
            yield Violation(pointer, f"{value} is below the minimum {schema['minimum']}")

    def _maximum(self, value: object, schema: Schema, pointer: str) -> Iterator[Violation]:
        if _is_type(value, "number") and value > schema["maximum"]:  # type: ignore[operator]
            yield Violation(pointer, f"{value} is above the maximum {schema['maximum']}")

    def _one_of(self, value: object, schema: Schema, pointer: str) -> Iterator[Violation]:
        matches = sum(not self.errors(value, sub, pointer) for sub in schema["oneOf"])  # type: ignore[union-attr]
        if matches != 1:
            yield Violation(pointer, f"matches {matches} of the {len(schema['oneOf'])} "  # type: ignore[arg-type]
                            "allowed shapes; exactly one is required")

    def _if(self, value: object, schema: Schema, pointer: str) -> Iterator[Violation]:
        if not self.errors(value, schema["if"], pointer) and "then" in schema:  # type: ignore[arg-type]
            yield from self._validate(value, schema["then"], pointer)  # type: ignore[arg-type]


def _nothing(value: object, schema: Schema, pointer: str) -> Iterator[Violation]:
    return iter(())  # "then" is applied by "if"


def _equal(a: object, b: object) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    return a == b


@cache
def _ecma_pattern(pattern: str) -> re.Pattern[str]:
    """A schema ``pattern`` with ECMA-262 meanings that Python's ``re`` differs on.

    ``$`` matches only at the very end (Python's also matches before a final
    newline), and ``\\d``, ``\\w``, ``\\s`` are ASCII classes. Only an unescaped
    ``$`` at the end of the pattern is rewritten, which covers the anchored
    patterns the shipped contracts use.
    """
    if pattern.endswith("$") and not pattern.endswith("\\$"):
        pattern = pattern[:-1] + r"\Z"
    return re.compile(pattern, re.ASCII)


def _kind(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, (list, tuple)):
        return "array"
    return type(value).__name__


def _short(value: object) -> str:
    return repr(value)[:80]
