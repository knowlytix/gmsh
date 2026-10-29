# SPDX-License-Identifier: Apache-2.0
"""Audit configuration and shipped profiles.

A profile (``schemas/profiles/<name>.json``) is a versioned list of rules, the
rules a gate requires by default, and parameter defaults. ``AuditConfig``
selects a profile and view and may narrow the rules, disable some, override
parameters or severities, and may name an audit schema sidecar file. A serialized
configuration holds data only: rule ids, parameter values, severities and a
path, never code to import.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cache
from importlib.resources import files
from types import MappingProxyType

from knowlytix.knowledge.kgcheck.frozen import FrozenJson, freeze_mapping, thaw
from knowlytix.knowledge.kgcheck.models import GraphView, JsonObject, Severity

DEFAULT_PROFILE = "ontodisc_default"
DEFAULT_MAX_FINDINGS = 200


class ConfigError(ValueError):
    """An invalid profile, rule selection or parameter."""


@dataclass(frozen=True)
class Profile:
    name: str
    version: str
    description: str
    rules: tuple[str, ...]
    required_rules: tuple[str, ...]
    params: Mapping[str, Mapping[str, FrozenJson]]


def _profile_dir():  # type: ignore[no-untyped-def]
    return files("knowlytix.knowledge.kgcheck").joinpath("schemas", "profiles")


def profile_names() -> tuple[str, ...]:
    """Names of the shipped profiles."""
    return tuple(sorted(p.name.removesuffix(".json") for p in _profile_dir().iterdir()
                        if p.name.endswith(".json")))


@cache
def load_profile(name: str) -> Profile:
    """A shipped profile by name. Raises ConfigError for an unknown name."""
    if name not in profile_names():
        raise ConfigError(f"unknown profile {name!r}; shipped profiles: {', '.join(profile_names())}")
    return parse_profile(json.loads(_profile_dir().joinpath(f"{name}.json").read_text(encoding="utf-8")))


def parse_profile(data: Mapping[str, object]) -> Profile:
    """A profile from its JSON form. Raises ConfigError when a rule is listed twice."""
    rules = list(data["rules"])  # type: ignore[call-overload]
    repeated = sorted({r for r in rules if rules.count(r) > 1})
    if repeated:
        raise ConfigError(f"profile {data['profile']!r} lists rules more than once: "
                          f"{', '.join(repeated)}")
    params = MappingProxyType({k: freeze_mapping(v) for k, v in data.get("params", {}).items()})
    return Profile(data["profile"], str(data["version"]), data.get("description", ""),
                   tuple(data["rules"]), tuple(data.get("required_rules", ())), params)


def _strings(value: Iterable[str] | None, key: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{key} must be a list of rule ids, got {_kind(value)}")
    return tuple(value)


@dataclass(frozen=True)
class AuditConfig:
    """How to run an audit.

    ``rules`` narrows the profile's rules (None keeps all of them); ``disabled``
    rules are recorded with status ``disabled``. ``params`` and ``severity`` are
    keyed by rule id. At most ``max_findings`` findings are kept per rule, the most
    severe first; the exact count is kept in the rule's metrics. ``audit_schema``
    is a sidecar file used instead of the store's own ``audit_schema.json``.
    """

    profile: str = DEFAULT_PROFILE
    view: GraphView = GraphView.STORED
    rules: tuple[str, ...] | None = None
    disabled: tuple[str, ...] = ()
    params: Mapping[str, Mapping[str, FrozenJson]] = field(default_factory=dict)
    severity: Mapping[str, Severity] = field(default_factory=dict)
    max_findings: int = DEFAULT_MAX_FINDINGS
    audit_schema: str | None = None
    # reviewed findings to keep visible but not block on; see ``check_suppression``
    suppressions: tuple[Mapping[str, FrozenJson], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.profile, str):
            raise ConfigError(f"profile must be a profile name, got {_kind(self.profile)}")
        if self.audit_schema is not None and not isinstance(self.audit_schema, str):
            raise ConfigError(f"audit_schema must be a file path, got {_kind(self.audit_schema)}")
        object.__setattr__(self, "view", _view(self.view))
        object.__setattr__(self, "rules", _strings(self.rules, "rules"))
        object.__setattr__(self, "disabled", _strings(self.disabled, "disabled") or ())
        params, severity = _object(self.params, "params"), _object(self.severity, "severity")
        object.__setattr__(self, "params", MappingProxyType(
            {str(k): freeze_mapping(_object(v, f"params.{k}")) for k, v in params.items()}))
        object.__setattr__(self, "severity", MappingProxyType(
            {str(k): _severity(v) for k, v in severity.items()}))
        if isinstance(self.max_findings, bool) or not isinstance(self.max_findings, int) \
                or self.max_findings < 1:
            raise ConfigError("max_findings must be a positive integer")
        if not isinstance(self.suppressions, (list, tuple)):
            raise ConfigError("suppressions must be a list of objects")
        object.__setattr__(self, "suppressions", tuple(
            freeze_mapping(check_suppression(s)) for s in self.suppressions))

    def to_dict(self) -> JsonObject:
        """Plain data; ``from_dict`` restores an equal configuration."""
        return {"profile": self.profile, "view": self.view.value,
                "rules": list(self.rules) if self.rules is not None else None,
                "disabled": list(self.disabled),
                "params": {k: thaw(v) for k, v in sorted(self.params.items())},
                "severity": {k: v.value for k, v in sorted(self.severity.items())},
                "max_findings": self.max_findings, "audit_schema": self.audit_schema,
                "suppressions": [thaw(s) for s in self.suppressions]}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> AuditConfig:
        """Configuration from plain data. Raises ConfigError on unknown keys or values."""
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ConfigError(f"unknown configuration keys: {', '.join(unknown)}")
        try:
            return cls(**data)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ConfigError(str(exc)) from exc


_SUPPRESSION_KEYS = {"rule_id", "rule_version", "reason", "finding_id", "affected",
                     "discriminator", "view"}


def check_suppression(value: object) -> Mapping[str, object]:
    """A suppression: ``rule_id``, the ``rule_version`` it was reviewed under, a nonblank
    ``reason``, and the finding by ``finding_id`` or by its ``affected`` ids, optionally
    limited to one ``view``. ``affected`` alone matches every finding of the rule on those
    ids; add ``discriminator`` to match one kind of issue. Raises ConfigError otherwise."""
    if not isinstance(value, Mapping):
        raise ConfigError(f"a suppression must be an object, got {_kind(value)}")
    unknown = sorted(set(value) - _SUPPRESSION_KEYS)
    if unknown:
        raise ConfigError(f"unknown suppression keys: {', '.join(unknown)}")
    for key in ("rule_id", "rule_version", "reason"):
        if not isinstance(value.get(key), str) or not str(value[key]).strip():
            raise ConfigError(f"a suppression needs a nonblank {key}")
    if ("finding_id" in value) == ("affected" in value):
        raise ConfigError("a suppression names exactly one of finding_id and affected")
    affected = value.get("affected")
    if affected is not None and (not isinstance(affected, (list, tuple)) or not affected
                                 or not all(isinstance(a, str) for a in affected)):
        raise ConfigError("suppression affected must be a nonempty list of ids")
    if "view" in value:
        _view(value["view"])
    if "discriminator" in value and ("finding_id" in value or not isinstance(value["discriminator"], str)):
        raise ConfigError("a suppression discriminator is a string used with affected")
    return value


def _view(value: object) -> GraphView:
    try:
        return GraphView(value)
    except ValueError as exc:
        raise ConfigError(f"unknown view {value!r}; views: "
                          f"{', '.join(v.value for v in GraphView)}") from exc


def _severity(value: object) -> Severity:
    try:
        return Severity(value)
    except ValueError as exc:
        raise ConfigError(f"unknown severity {value!r}; severities: "
                          f"{', '.join(s.value for s in Severity)}") from exc


def _object(value: object, key: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} must be a JSON object keyed by rule id or parameter, "
                          f"got {_kind(value)}")
    return value


def _kind(value: object) -> str:
    return "null" if value is None else type(value).__name__


@dataclass(frozen=True)
class Plan:
    """The resolved rule selection of one audit."""

    run: tuple[str, ...]
    disabled: tuple[str, ...]
    params: Mapping[str, Mapping[str, FrozenJson]]


def resolve(config: AuditConfig, specs: Mapping[str, object]) -> Plan:
    """Rules to run and their parameters, validated against the registered specs.

    ``specs`` maps rule id to a ``RuleSpec``. Raises ConfigError for a rule the
    registry lacks, a parameter a rule does not declare, or a value of the
    wrong type.
    """
    profile = load_profile(config.profile)
    selected = _select(config, profile, specs)
    named = (set(config.disabled) | set(config.params) | set(config.severity)
             | {str(s["rule_id"]) for s in config.suppressions})
    _known(named, specs)
    params = {rid: _params(rid, specs[rid], profile.params.get(rid, {}), config.params.get(rid, {}))
              for rid in selected}
    return Plan(tuple(r for r in selected if r not in config.disabled),
                tuple(r for r in selected if r in config.disabled), MappingProxyType(params))


def _select(config: AuditConfig, profile: Profile, specs: Mapping[str, object]) -> tuple[str, ...]:
    if config.rules is None:
        _known(set(profile.rules), specs, where=f"profile {profile.name}")
        return tuple(sorted(profile.rules))
    _known(set(config.rules), specs)
    return tuple(sorted(set(config.rules)))


def _known(ids: set[str], specs: Mapping[str, object], where: str = "configuration") -> None:
    unknown = sorted(ids - set(specs))
    if unknown:
        raise ConfigError(f"{where} names unregistered rules: {', '.join(unknown)}")


def _params(rule_id: str, spec: object, *layers: Mapping[str, FrozenJson]) -> Mapping[str, FrozenJson]:
    defaults: Mapping[str, FrozenJson] = getattr(spec, "parameters")
    merged = dict(defaults)
    for layer in layers:
        for key, value in layer.items():
            if key not in defaults:
                raise ConfigError(f"{rule_id} has no parameter {key!r}; parameters: "
                                  f"{', '.join(sorted(defaults)) or 'none'}")
            if not _same_type(defaults[key], value):
                raise ConfigError(f"{rule_id}.{key} must be {type(defaults[key]).__name__}")
            _in_bounds(rule_id, key, value, getattr(spec, "parameter_bounds", {}).get(key))
            _in_choices(rule_id, key, value, getattr(spec, "parameter_choices", {}).get(key))
            merged[key] = value
    for low, high in getattr(spec, "parameter_order", ()):
        if merged[low] > merged[high]:  # type: ignore[operator]
            raise ConfigError(f"{rule_id}.{low} ({merged[low]}) exceeds {rule_id}.{high} "
                              f"({merged[high]})")
    return MappingProxyType(merged)


def _in_bounds(rule_id: str, key: str, value: FrozenJson,
               bounds: tuple[float | None, float | None] | None) -> None:
    """Reject a non-finite number, or one outside the rule's declared bounds."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    if not math.isfinite(value):
        raise ConfigError(f"{rule_id}.{key} must be a finite number")
    low, high = bounds or (None, None)
    if (low is not None and value < low) or (high is not None and value > high):
        raise ConfigError(f"{rule_id}.{key} must lie in [{low}, {high}], got {value}")


def _in_choices(rule_id: str, key: str, value: FrozenJson, choices: tuple[str, ...] | None) -> None:
    """Reject a text value, or a list item, that the rule does not offer."""
    if choices is None:
        return
    items = value if isinstance(value, (list, tuple)) else (value,)
    if not items:
        raise ConfigError(f"{rule_id}.{key} needs at least one of {', '.join(choices)}")
    bad = [v for v in items if v not in choices]
    if bad:
        raise ConfigError(f"{rule_id}.{key} accepts {', '.join(choices)}; got {', '.join(map(str, bad))}")


def _same_type(default: FrozenJson, value: FrozenJson) -> bool:
    if isinstance(default, bool) or isinstance(value, bool):
        return isinstance(default, bool) and isinstance(value, bool)
    if isinstance(default, float):
        return isinstance(value, (int, float))
    if isinstance(default, tuple):  # items take the type of the default's items (text if empty)
        item = type(default[0]) if default else str
        return isinstance(value, (tuple, list)) and all(
            isinstance(v, item) and not isinstance(v, bool) for v in value)
    return isinstance(value, type(default))
