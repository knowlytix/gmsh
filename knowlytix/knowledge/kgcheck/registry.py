# SPDX-License-Identifier: Apache-2.0
"""Rule declarations, the read-only context a rule runs in, and the registry.

A rule is a ``RuleSpec`` (stable id, version, category, prerequisites, views,
parameters) plus a check function. The check receives a ``RuleContext`` and
returns an ``Outcome``: its findings and the population it examined. The engine,
not the rule, turns an outcome into a ``RuleResult`` status, so every rule obeys
the same status semantics. Rules are registered as Python objects only; nothing
in a serialized configuration can name code to import.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from knowlytix.knowledge.kgcheck.config import ConfigError
from knowlytix.knowledge.kgcheck.frozen import FrozenJson, freeze_mapping
from knowlytix.knowledge.kgcheck.models import (
    AuditSnapshot,
    Finding,
    GraphView,
    RawRef,
    Severity,
)

# A prerequisite names what a rule needs from the snapshot. Capability names are
# the boolean fields of ``Capabilities``; the others are resolved by the engine.
VIEW = "view"  # the selected graph view is present
ONTOLOGY = "ontology"  # ontology.json parsed as an object


def artifact_prerequisite(name: str) -> str:
    """The prerequisite that artifact ``name`` parsed."""
    return f"artifact:{name}"


@dataclass(frozen=True)
class RuleSpec:
    """What a rule is, independent of how it is implemented.

    ``views`` is None for a rule that does not depend on the selected view;
    otherwise the views it can audit. ``parameters`` holds every parameter the
    rule accepts with its default; a configuration may only override these.
    """

    rule_id: str
    version: str
    category: str
    title: str
    default_severity: Severity
    phase: str = "A"
    prerequisites: tuple[str, ...] = ()
    views: tuple[GraphView, ...] | None = None
    parameters: Mapping[str, FrozenJson] = field(default_factory=dict)
    # inclusive (low, high) for numeric parameters; None leaves that side open
    parameter_bounds: Mapping[str, tuple[float | None, float | None]] = field(default_factory=dict)
    # allowed values of a text parameter, or of each item of a list parameter
    parameter_choices: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    # (low, high) parameter pairs whose values must satisfy low <= high
    parameter_order: tuple[tuple[str, str], ...] = ()
    basis: str = ""  # detection method, quoted in every finding
    references: tuple[str, ...] = ()
    # False when the result depends on more than the snapshot and the configuration
    # (producer code, a file named by a parameter, an external engine or model), so an
    # incremental audit must always run the rule (kgcheck.incremental)
    cacheable: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "prerequisites", tuple(self.prerequisites))
        object.__setattr__(self, "references", tuple(self.references))
        if self.views is not None:
            object.__setattr__(self, "views", tuple(GraphView(v) for v in self.views))
        object.__setattr__(self, "parameters", freeze_mapping(self.parameters))
        object.__setattr__(self, "parameter_bounds", MappingProxyType(
            {k: (lo, hi) for k, (lo, hi) in self.parameter_bounds.items()}))
        if any(isinstance(v, str) for v in self.parameter_choices.values()):
            raise ValueError(f"{self.rule_id}: parameter choices must be a tuple of strings")
        object.__setattr__(self, "parameter_choices", MappingProxyType(
            {k: tuple(v) for k, v in self.parameter_choices.items()}))
        object.__setattr__(self, "parameter_order", tuple(tuple(p) for p in self.parameter_order))
        named = {*self.parameter_bounds, *self.parameter_choices,
                 *(k for pair in self.parameter_order for k in pair)}
        unknown = sorted(named - set(self.parameters))
        if unknown:
            raise ValueError(f"{self.rule_id}: constraints for undeclared parameters {unknown}")
        for key, choices in self.parameter_choices.items():
            default = self.parameters[key]
            items = default if isinstance(default, tuple) else (default,)
            if any(v not in choices for v in items):
                raise ValueError(f"{self.rule_id}: default of {key} is not among its choices")


class OutcomeError(ValueError):
    """An outcome that contradicts itself."""


@dataclass(frozen=True)
class Outcome:
    """What a check found and how much it examined.

    ``eligible`` targets existed and ``tested`` of them were examined (None means
    all of them). ``missing`` names prerequisites discovered only while running:
    the result is then unavailable and carries no findings. ``partial`` explains
    an execution that stopped before examining every eligible target.
    """

    eligible: int
    findings: tuple[Finding, ...] = ()
    tested: int | None = None
    metrics: Mapping[str, FrozenJson] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    partial: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "findings", tuple(self.findings))
        object.__setattr__(self, "missing", tuple(self.missing))
        object.__setattr__(self, "metrics", freeze_mapping(self.metrics))
        if self.missing and self.findings:
            raise OutcomeError("an outcome with missing prerequisites carries no findings")

    @property
    def examined(self) -> int:
        return self.eligible if self.tested is None else self.tested

    @classmethod
    def unavailable(cls, *missing: str) -> Outcome:
        return cls(eligible=0, missing=missing)


def ratio(numerator: int, denominator: int) -> dict[str, int | float | None]:
    """A metric with its denominator; the value is None when nothing was counted."""
    value = round(numerator / denominator, 6) if denominator else None
    return {"numerator": numerator, "denominator": denominator, "value": value}


@dataclass(frozen=True)
class RuleContext:
    """The read-only input of one rule execution."""

    snapshot: AuditSnapshot
    view: GraphView
    spec: RuleSpec
    params: Mapping[str, FrozenJson]

    def finding(self, message: str, *, severity: Severity | None = None,
                affected: Iterable[str] = (), locations: Iterable[RawRef] = (),
                witness: Mapping[str, object] | None = None, suggestion: str | None = None,
                discriminator: str = "", view: GraphView | None = None,
                view_scoped: bool = True) -> Finding:
        """A finding of this rule; the view defaults to the selected one for view rules."""
        scoped = view or (self.view if view_scoped and self.spec.views is not None else None)
        return Finding(rule_id=self.spec.rule_id, rule_version=self.spec.version,
                       category=self.spec.category, severity=severity or self.spec.default_severity,
                       message=message, basis=self.spec.basis, view=scoped,
                       affected=tuple(affected), locations=tuple(locations),
                       witness=witness or {}, suggestion=suggestion, discriminator=discriminator)


RuleCheck = Callable[[RuleContext], Outcome]


@dataclass(frozen=True)
class Rule:
    spec: RuleSpec
    check: RuleCheck

    @property
    def rule_id(self) -> str:
        return self.spec.rule_id


class DuplicateRuleError(ConfigError):
    """Two rules registered under one id."""


class RuleRegistry:
    """Rules by id. ``builtin_registry()`` returns the shipped rules."""

    def __init__(self, rules: Iterable[Rule] = ()) -> None:
        self._rules: dict[str, Rule] = {}
        for rule in rules:
            self.register(rule)

    def register(self, rule: Rule) -> Rule:
        if rule.rule_id in self._rules:
            raise DuplicateRuleError(f"rule {rule.rule_id} is already registered")
        self._rules[rule.rule_id] = rule
        return rule

    def get(self, rule_id: str) -> Rule:
        return self._rules[rule_id]

    def __contains__(self, rule_id: object) -> bool:
        return rule_id in self._rules

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._rules))

    def rules(self) -> Mapping[str, Rule]:
        return MappingProxyType(dict(sorted(self._rules.items())))

    def with_rules(self, rules: Iterable[Rule]) -> RuleRegistry:
        """A new registry holding these rules and the extra ones."""
        return RuleRegistry([*self._rules.values(), *rules])


_BUILTIN: list[Rule] = []


def builtin(spec: RuleSpec) -> Callable[[RuleCheck], RuleCheck]:
    """Decorator registering a shipped rule; used only inside ``kgcheck.rules``."""
    def register(check: RuleCheck) -> RuleCheck:
        _BUILTIN.append(Rule(spec, check))
        return check
    return register


def builtin_registry() -> RuleRegistry:
    """A fresh registry of every shipped rule."""
    import knowlytix.knowledge.kgcheck.rules  # noqa: F401  # registers on import
    return RuleRegistry(_BUILTIN)
