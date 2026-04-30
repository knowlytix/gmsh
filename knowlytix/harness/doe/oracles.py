# SPDX-License-Identifier: Apache-2.0
"""GMSH Oracle Framework — composite oracles for black-box agentic testing.

The hardest part of black-box testing is the oracle: how do we judge
whether SUT behavior is acceptable when we cannot see its internals?
This module provides a pluggable oracle framework with seven oracle types,
each leveraging GMS geometric primitives where applicable.

Oracle types:

- **ExactOutcomeOracle** — single correct answer (score_triple)
- **AcceptableSetOracle** — set of valid outputs (geodesic to nearest)
- **ConstraintOracle** — behavioral constraints (must/must-not predicates)
- **TraceOracle** — tool-call sequence patterns (check_holonomy)
- **SideEffectOracle** — environment state changes (tension_energy)
- **HumanReviewedOracle** — rubric-driven LLM/human review
- **CompositeOracle** — weighted combination of the above

Usage::

    oracle = CompositeOracle(oracles=[
        (ExactOutcomeOracle(store=store), 0.4),
        (TraceOracle(store=store), 0.3),
        (ConstraintOracle(store=store), 0.3),
    ])
    verdict = oracle.evaluate(scenario, trace, env_state)
"""

from __future__ import annotations

import fnmatch
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from knowlytix.doc.store import GMSExpertStore
    from .adapters import SUTTrace
    from .environment import EnvironmentSimulator
    from .scenarios import Scenario
    from .semantic import RubricDimension


# ---------------------------------------------------------------------------
# Oracle verdict
# ---------------------------------------------------------------------------

@dataclass
class OracleVerdict:
    """Structured result from an oracle evaluation.

    Attributes
    ----------
    oracle_type : str
        Which oracle type produced this verdict.
    passed : bool
        Whether the SUT behavior is acceptable.
    confidence : float
        [0, 1] confidence in the verdict.
    detail : str
        Human-readable explanation.
    evidence : list[str]
        Supporting evidence items (quotes, tool names, state keys).
    raw_scores : dict[str, float]
        Named scores from the oracle internals.
    """

    oracle_type: str = ""
    passed: bool = False
    confidence: float = 0.0
    detail: str = ""
    evidence: list[str] = field(default_factory=list)
    raw_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "oracle_type": self.oracle_type,
            "passed": self.passed,
            "confidence": round(self.confidence, 4),
            "detail": self.detail,
            "evidence": list(self.evidence),
            "raw_scores": {k: round(v, 4) for k, v in self.raw_scores.items()},
        }


# ---------------------------------------------------------------------------
# Base oracle
# ---------------------------------------------------------------------------

class OracleBase(ABC):
    """Abstract base for all oracle types."""

    oracle_type: str = "base"

    @abstractmethod
    def evaluate(
        self,
        scenario: Scenario,
        trace: SUTTrace,
        env_state_before: dict | None = None,
        env_state_after: dict | None = None,
    ) -> OracleVerdict:
        """Evaluate SUT behavior and return a verdict.

        Parameters
        ----------
        scenario : Scenario
            The test scenario definition including expected behavior.
        trace : SUTTrace
            Captured SUT execution trace (output, tool calls, timing).
        env_state_before : dict or None
            Environment state snapshot before SUT execution.
        env_state_after : dict or None
            Environment state snapshot after SUT execution.
        """
        ...


# ---------------------------------------------------------------------------
# Oracle implementations
# ---------------------------------------------------------------------------

class ExactOutcomeOracle(OracleBase):
    """Oracle for cases with one correct final answer.

    Uses string comparison and optionally GMS ``score_triple()`` to
    verify that the SUT output matches the expected ground truth.
    """

    oracle_type = "exact_outcome"

    def __init__(self, store: GMSExpertStore | None = None):
        self._store = store

    def evaluate(
        self, scenario, trace, env_state_before=None, env_state_after=None,
    ) -> OracleVerdict:
        expected = str(scenario.ground_truth).strip().lower()
        actual = str(trace.raw_output).strip().lower()

        if not expected:
            return OracleVerdict(
                oracle_type=self.oracle_type,
                passed=True,
                confidence=0.5,
                detail="No ground truth defined; skipping exact check.",
            )

        # Direct string match
        exact_match = expected in actual or actual in expected

        # GMS geometric score
        gms_score = None
        if self._store is not None and scenario.ground_truth:
            try:
                gms_score = self._store.score_triple(
                    "answer", "equals", scenario.ground_truth,
                )
            except Exception:
                pass

        confidence = 1.0 if exact_match else 0.0
        if gms_score is not None:
            # Lower geodesic = better match
            geo_conf = max(0.0, 1.0 - gms_score / 2.0)
            confidence = max(confidence, geo_conf)

        passed = exact_match or confidence >= 0.7

        return OracleVerdict(
            oracle_type=self.oracle_type,
            passed=passed,
            confidence=confidence,
            detail=f"exact_match={exact_match}, gms_score={gms_score}",
            evidence=[f"expected={scenario.ground_truth}", f"actual={trace.raw_output[:200]}"],
            raw_scores={"exact_match": 1.0 if exact_match else 0.0,
                        **({"gms_geodesic": gms_score} if gms_score is not None else {})},
        )


class AcceptableSetOracle(OracleBase):
    """Oracle for cases where several outputs are valid.

    Checks if the SUT output matches any item in an acceptable set,
    using string containment and GMS geodesic distance to the nearest
    acceptable answer.
    """

    oracle_type = "acceptable_set"

    def __init__(self, store: GMSExpertStore | None = None):
        self._store = store

    def evaluate(
        self, scenario, trace, env_state_before=None, env_state_after=None,
    ) -> OracleVerdict:
        acceptable = scenario.metadata.get("acceptable_set", [])
        if not acceptable:
            # Fall back to ground_truth as single-item set
            if scenario.ground_truth:
                acceptable = [scenario.ground_truth]
            else:
                return OracleVerdict(
                    oracle_type=self.oracle_type, passed=True,
                    confidence=0.5, detail="No acceptable set defined.",
                )

        actual = str(trace.raw_output).strip().lower()
        best_match = None
        best_score = 0.0

        for candidate in acceptable:
            candidate_lower = str(candidate).strip().lower()
            if candidate_lower in actual or actual in candidate_lower:
                best_match = candidate
                best_score = 1.0
                break

            # GMS geodesic distance
            if self._store is not None:
                try:
                    geo = self._store.score_triple("answer", "equals", str(candidate))
                    score = max(0.0, 1.0 - geo / 2.0)
                    if score > best_score:
                        best_score = score
                        best_match = candidate
                except Exception:
                    pass

        passed = best_score >= 0.7

        return OracleVerdict(
            oracle_type=self.oracle_type,
            passed=passed,
            confidence=best_score,
            detail=f"best_match={best_match}, score={best_score:.3f}, "
                   f"checked {len(acceptable)} candidates",
            evidence=[f"acceptable_set_size={len(acceptable)}",
                      f"best_match={best_match}"],
            raw_scores={"best_score": best_score},
        )


class ConstraintOracle(OracleBase):
    """Oracle for behavioral constraint satisfaction.

    Evaluates predicates on the SUT output, trace, and environment state.

    Supported constraint types:

    - ``must_contain`` — output must contain this string
    - ``must_not_contain`` — output must not contain this string
    - ``max_tool_calls`` — tool call budget
    - ``max_latency_ms`` — timing budget
    - ``required_tools`` — tools that must appear in trace
    - ``forbidden_tools`` — tools that must not appear in trace
    - ``output_max_length`` — character limit on output
    - ``state_invariant`` — environment state key must not change
    """

    oracle_type = "constraint"

    def __init__(self, store: GMSExpertStore | None = None):
        self._store = store

    def evaluate(
        self, scenario, trace, env_state_before=None, env_state_after=None,
    ) -> OracleVerdict:
        constraints = scenario.constraints if hasattr(scenario, "constraints") else []
        if not constraints:
            return OracleVerdict(
                oracle_type=self.oracle_type, passed=True,
                confidence=1.0, detail="No constraints defined.",
            )

        violations = []
        checks_passed = 0
        total_checks = len(constraints)

        for c in constraints:
            ctype = c.get("type", "")
            value = c.get("value", "")

            if ctype == "must_contain":
                if value.lower() not in trace.raw_output.lower():
                    violations.append(f"Output missing required: '{value}'")
                else:
                    checks_passed += 1

            elif ctype == "must_not_contain":
                if value.lower() in trace.raw_output.lower():
                    violations.append(f"Output contains prohibited: '{value}'")
                else:
                    checks_passed += 1

            elif ctype == "max_tool_calls":
                actual = len(trace.tool_calls) if trace.tool_calls else 0
                limit = int(value)
                if actual > limit:
                    violations.append(
                        f"Tool calls {actual} exceeds limit {limit}"
                    )
                else:
                    checks_passed += 1

            elif ctype == "max_latency_ms":
                limit = float(value)
                if trace.latency_ms > limit:
                    violations.append(
                        f"Latency {trace.latency_ms:.0f}ms exceeds {limit:.0f}ms"
                    )
                else:
                    checks_passed += 1

            elif ctype == "required_tools":
                tools_used = {
                    tc.tool_name for tc in (trace.tool_calls or [])
                }
                required = set(value) if isinstance(value, list) else {value}
                missing = required - tools_used
                if missing:
                    violations.append(f"Required tools missing: {missing}")
                else:
                    checks_passed += 1

            elif ctype == "forbidden_tools":
                tools_used = {
                    tc.tool_name for tc in (trace.tool_calls or [])
                }
                forbidden = set(value) if isinstance(value, list) else {value}
                found = tools_used & forbidden
                if found:
                    violations.append(f"Forbidden tools used: {found}")
                else:
                    checks_passed += 1

            elif ctype == "output_max_length":
                limit = int(value)
                if len(trace.raw_output) > limit:
                    violations.append(
                        f"Output length {len(trace.raw_output)} exceeds {limit}"
                    )
                else:
                    checks_passed += 1

            elif ctype == "state_invariant":
                if env_state_before and env_state_after:
                    before_val = env_state_before.get(value)
                    after_val = env_state_after.get(value)
                    if before_val != after_val:
                        violations.append(
                            f"State '{value}' changed: {before_val} → {after_val}"
                        )
                    else:
                        checks_passed += 1
                else:
                    checks_passed += 1  # can't check without state

            else:
                checks_passed += 1  # unknown constraint type → pass

        passed = len(violations) == 0
        confidence = checks_passed / max(total_checks, 1)

        return OracleVerdict(
            oracle_type=self.oracle_type,
            passed=passed,
            confidence=confidence,
            detail=f"{checks_passed}/{total_checks} constraints satisfied",
            evidence=violations if violations else ["All constraints satisfied"],
            raw_scores={"checks_passed": float(checks_passed),
                        "total_checks": float(total_checks),
                        "violations": float(len(violations))},
        )


class TraceOracle(OracleBase):
    """Oracle for tool-call sequence verification.

    Verifies that the SUT's tool-call sequence conforms to an expected
    pattern.  Supports:

    - Required steps (must appear in trace)
    - Prohibited subsequences (must NOT appear)
    - Ordering constraints (A must come before B)
    - GMS holonomy check (geometric process deviation)

    The trace pattern is a comma-separated list of tool names with
    optional wildcards (``*``).  Prefix with ``!`` for prohibited.
    Suffix with ``>`` for ordering constraints::

        "gms_lookup, gms_search, !payment_execute"
        "verify_identity > approve_refund"
    """

    oracle_type = "trace"

    def __init__(self, store: GMSExpertStore | None = None):
        self._store = store

    def evaluate(
        self, scenario, trace, env_state_before=None, env_state_after=None,
    ) -> OracleVerdict:
        pattern = getattr(scenario, "trace_pattern", "")
        if not pattern:
            # Fall back to expected_tool_sequence from Scenario
            if scenario.expected_tool_sequence:
                return self._check_sequence(scenario, trace)
            return OracleVerdict(
                oracle_type=self.oracle_type, passed=True,
                confidence=0.5, detail="No trace pattern defined.",
            )

        actual_tools = [
            tc.tool_name for tc in (trace.tool_calls or [])
        ]
        violations = []
        checks = 0
        checks_passed = 0

        # Parse pattern elements
        elements = [e.strip() for e in pattern.split(",")]
        for elem in elements:
            elem = elem.strip()
            if not elem:
                continue

            # Ordering constraint: "A > B"
            if ">" in elem:
                parts = [p.strip() for p in elem.split(">")]
                checks += 1
                order_ok = True
                for i in range(len(parts) - 1):
                    a, b = parts[i], parts[i + 1]
                    idx_a = self._find_tool_index(actual_tools, a)
                    idx_b = self._find_tool_index(actual_tools, b)
                    if idx_a is None:
                        violations.append(f"Required tool '{a}' not in trace")
                        order_ok = False
                    elif idx_b is None:
                        violations.append(f"Required tool '{b}' not in trace")
                        order_ok = False
                    elif idx_a >= idx_b:
                        violations.append(
                            f"Ordering violation: '{a}' (idx={idx_a}) "
                            f"must come before '{b}' (idx={idx_b})"
                        )
                        order_ok = False
                if order_ok:
                    checks_passed += 1

            # Prohibited: "!tool_name"
            elif elem.startswith("!"):
                tool_pat = elem[1:].strip()
                checks += 1
                found_prohibited = False
                for t in actual_tools:
                    if fnmatch.fnmatch(t, tool_pat):
                        violations.append(f"Prohibited tool found: '{t}'")
                        found_prohibited = True
                        break
                if not found_prohibited:
                    checks_passed += 1

            # Required: "tool_name"
            else:
                checks += 1
                found = any(
                    fnmatch.fnmatch(t, elem) for t in actual_tools
                )
                if not found:
                    violations.append(f"Required tool not found: '{elem}'")
                else:
                    checks_passed += 1

        # GMS holonomy check if store available and expected sequence defined
        holonomy_score = None
        if (self._store is not None
                and scenario.expected_tool_sequence
                and actual_tools):
            holonomy_score = self._check_holonomy(
                scenario.expected_tool_sequence, actual_tools,
            )

        passed = len(violations) == 0
        confidence = (checks - len(violations)) / max(checks, 1)

        raw_scores = {
            "checks": float(checks),
            "violations": float(len(violations)),
        }
        if holonomy_score is not None:
            raw_scores["holonomy_defect"] = holonomy_score
            # High holonomy defect reduces confidence
            if holonomy_score > 0.5:
                confidence *= max(0.5, 1.0 - holonomy_score)

        return OracleVerdict(
            oracle_type=self.oracle_type,
            passed=passed,
            confidence=confidence,
            detail=f"trace: {len(actual_tools)} calls, {len(violations)} violations",
            evidence=violations if violations else [
                f"Trace conforms: {actual_tools[:5]}..."
            ],
            raw_scores=raw_scores,
        )

    def _check_sequence(self, scenario, trace) -> OracleVerdict:
        """Check expected_tool_sequence against actual."""
        expected = scenario.expected_tool_sequence
        actual = [tc.tool_name for tc in (trace.tool_calls or [])]
        violations = []

        for tool in expected:
            if tool not in actual:
                violations.append(f"Expected tool missing: '{tool}'")

        passed = len(violations) == 0
        return OracleVerdict(
            oracle_type=self.oracle_type,
            passed=passed,
            confidence=1.0 if passed else 0.0,
            detail=f"expected {len(expected)} tools, {len(violations)} missing",
            evidence=violations or ["All expected tools found"],
        )

    def _find_tool_index(
        self, tools: list[str], pattern: str,
    ) -> int | None:
        """Find the first index of a tool matching the pattern."""
        for i, t in enumerate(tools):
            if fnmatch.fnmatch(t, pattern):
                return i
        return None

    def _check_holonomy(
        self, expected: list[str], actual: list[str],
    ) -> float | None:
        """Compute holonomy defect between expected and actual paths."""
        try:
            return self._store.check_holonomy(expected, actual)
        except Exception:
            return None


class SideEffectOracle(OracleBase):
    """Oracle for environment state change verification.

    Compares environment state before and after SUT execution against
    expected side effects defined in the scenario.  Uses GMS
    ``tension_energy()`` to measure divergence between expected and
    actual state changes.

    Requires an :class:`EnvironmentSimulator` to have captured state
    snapshots.
    """

    oracle_type = "side_effect"

    def __init__(
        self,
        environment: EnvironmentSimulator | None = None,
        store: GMSExpertStore | None = None,
    ):
        self._environment = environment
        self._store = store

    def evaluate(
        self, scenario, trace, env_state_before=None, env_state_after=None,
    ) -> OracleVerdict:
        expected_effects = getattr(scenario, "expected_side_effects", {})
        if not expected_effects:
            return OracleVerdict(
                oracle_type=self.oracle_type, passed=True,
                confidence=0.5, detail="No expected side effects defined.",
            )

        if env_state_before is None or env_state_after is None:
            return OracleVerdict(
                oracle_type=self.oracle_type, passed=False,
                confidence=0.0,
                detail="Cannot evaluate: environment state not captured.",
            )

        # Compute actual diff
        if self._environment is not None:
            diff = self._environment.get_state_diff(
                env_state_before, env_state_after,
            )
        else:
            # Manual diff
            diff = {
                "added": {k: env_state_after[k]
                          for k in set(env_state_after) - set(env_state_before)},
                "removed": {k: env_state_before[k]
                            for k in set(env_state_before) - set(env_state_after)},
                "changed": {
                    k: {"before": env_state_before[k], "after": env_state_after[k]}
                    for k in set(env_state_before) & set(env_state_after)
                    if env_state_before[k] != env_state_after[k]
                },
            }

        violations = []
        checks_passed = 0
        total = len(expected_effects)

        for key, expected_val in expected_effects.items():
            actual_val = env_state_after.get(key)

            if expected_val == "__unchanged__":
                # State key must not have changed
                if key in diff.get("changed", {}):
                    violations.append(
                        f"State '{key}' changed unexpectedly: "
                        f"{diff['changed'][key]}"
                    )
                else:
                    checks_passed += 1

            elif expected_val == "__absent__":
                # State key must not exist
                if key in env_state_after:
                    violations.append(
                        f"State '{key}' should be absent but found: {actual_val}"
                    )
                else:
                    checks_passed += 1

            elif expected_val == "__present__":
                # State key must exist (any value)
                if key not in env_state_after:
                    violations.append(f"State '{key}' should be present")
                else:
                    checks_passed += 1

            else:
                # Exact value match
                if actual_val == expected_val:
                    checks_passed += 1
                else:
                    violations.append(
                        f"State '{key}': expected={expected_val}, "
                        f"actual={actual_val}"
                    )

        # GMS tension check between expected and actual state
        tension_score = None
        if self._store is not None and violations:
            try:
                tension_score = self._store.tension_energy(
                    "expected_state", "actual_state",
                )
            except Exception:
                pass

        passed = len(violations) == 0
        confidence = checks_passed / max(total, 1)

        raw_scores = {
            "checks_passed": float(checks_passed),
            "total_checks": float(total),
        }
        if tension_score is not None:
            raw_scores["tension_energy"] = tension_score

        return OracleVerdict(
            oracle_type=self.oracle_type,
            passed=passed,
            confidence=confidence,
            detail=f"{checks_passed}/{total} side effects verified",
            evidence=violations if violations else [
                "All expected side effects confirmed"
            ],
            raw_scores=raw_scores,
        )


class HumanReviewedOracle(OracleBase):
    """Oracle for rubric-driven evaluation of subjective quality.

    Uses a structured rubric (list of dimensions with descriptions and
    weights) to evaluate the SUT output.  Can be executed by an LLM
    judge or a human reviewer.  Reuses
    :class:`~harness.doe.semantic.RubricDimension`.

    For automated evaluation, an LLM is called with the rubric and
    asked to score each dimension 1-5.
    """

    oracle_type = "human_reviewed"

    def __init__(
        self,
        rubric: list[RubricDimension] | None = None,
        store: GMSExpertStore | None = None,
        llm_client: Any = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        self._rubric = rubric or []
        self._store = store
        self._client = llm_client
        self._model = model

    def evaluate(
        self, scenario, trace, env_state_before=None, env_state_after=None,
    ) -> OracleVerdict:
        if not self._rubric:
            return OracleVerdict(
                oracle_type=self.oracle_type, passed=True,
                confidence=0.5, detail="No rubric defined.",
            )

        # Build review prompt
        rubric_text = "\n".join(
            f"- {dim.name} (weight={dim.weight}): {dim.description}"
            for dim in self._rubric
        )
        prompt = (
            f"Evaluate this agent response using the rubric below.\n\n"
            f"Question: {scenario.question_text}\n"
            f"Expected: {scenario.ground_truth}\n"
            f"Agent response: {trace.raw_output[:2000]}\n\n"
            f"Rubric:\n{rubric_text}\n\n"
            f"Score each dimension 1-5 and provide brief reasoning.\n"
            f"Format: DIMENSION: SCORE/5 — reasoning"
        )

        if self._client is None:
            # No client — return a placeholder for manual review
            return OracleVerdict(
                oracle_type=self.oracle_type,
                passed=False,
                confidence=0.0,
                detail="Human review required (no LLM client configured).",
                evidence=[f"rubric_dimensions={len(self._rubric)}"],
            )

        # LLM judge call
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            judge_text = response.content[0].text
        except Exception as exc:
            return OracleVerdict(
                oracle_type=self.oracle_type,
                passed=False,
                confidence=0.0,
                detail=f"LLM judge failed: {exc}",
            )

        # Parse scores from judge response
        scores = {}
        total_weight = 0.0
        weighted_sum = 0.0
        for dim in self._rubric:
            pattern = rf"{re.escape(dim.name)}:\s*(\d)/5"
            match = re.search(pattern, judge_text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                scores[dim.name] = score
                weighted_sum += score * dim.weight
                total_weight += dim.weight

        avg_score = weighted_sum / max(total_weight, 1.0)
        confidence = avg_score / 5.0
        passed = avg_score >= 3.0

        return OracleVerdict(
            oracle_type=self.oracle_type,
            passed=passed,
            confidence=confidence,
            detail=f"LLM judge avg={avg_score:.2f}/5, {len(scores)}/{len(self._rubric)} dims scored",
            evidence=[f"{k}={v}/5" for k, v in scores.items()],
            raw_scores={k: float(v) for k, v in scores.items()},
        )


class CompositeOracle(OracleBase):
    """Weighted combination of multiple oracle types.

    Runs all sub-oracles and computes a weighted verdict.  A composite
    oracle fails if any sub-oracle with ``critical=True`` fails, or if
    the weighted confidence falls below the threshold.

    Parameters
    ----------
    oracles : list[tuple[OracleBase, float]]
        List of (oracle, weight) pairs.
    threshold : float
        Minimum weighted confidence to pass (default 0.5).
    fail_on_any_critical : bool
        If True, fail if any sub-oracle marked as critical fails.
    """

    oracle_type = "composite"

    def __init__(
        self,
        oracles: list[tuple[OracleBase, float]],
        threshold: float = 0.5,
        fail_on_any_critical: bool = True,
    ):
        self._oracles = oracles
        self._threshold = threshold
        self._fail_on_any_critical = fail_on_any_critical

    def evaluate(
        self, scenario, trace, env_state_before=None, env_state_after=None,
    ) -> OracleVerdict:
        sub_verdicts: list[OracleVerdict] = []
        total_weight = 0.0
        weighted_confidence = 0.0
        all_evidence: list[str] = []
        all_scores: dict[str, float] = {}
        any_critical_failed = False

        for oracle, weight in self._oracles:
            verdict = oracle.evaluate(
                scenario, trace, env_state_before, env_state_after,
            )
            sub_verdicts.append(verdict)
            total_weight += weight
            weighted_confidence += verdict.confidence * weight
            all_evidence.extend(verdict.evidence)

            # Prefix scores with oracle type
            for k, v in verdict.raw_scores.items():
                all_scores[f"{verdict.oracle_type}.{k}"] = v

            if not verdict.passed:
                any_critical_failed = True

        avg_confidence = weighted_confidence / max(total_weight, 1e-9)

        passed = avg_confidence >= self._threshold
        if self._fail_on_any_critical and any_critical_failed:
            passed = False

        # Build summary
        oracle_summaries = [
            f"{v.oracle_type}: {'PASS' if v.passed else 'FAIL'} "
            f"(conf={v.confidence:.3f})"
            for v in sub_verdicts
        ]

        return OracleVerdict(
            oracle_type=self.oracle_type,
            passed=passed,
            confidence=avg_confidence,
            detail=f"composite({len(self._oracles)} oracles): "
                   f"avg_conf={avg_confidence:.3f}, "
                   f"threshold={self._threshold} | "
                   + "; ".join(oracle_summaries),
            evidence=all_evidence[:20],  # cap evidence length
            raw_scores=all_scores,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_oracle(
    oracle_def: dict,
    store: GMSExpertStore | None = None,
    environment: EnvironmentSimulator | None = None,
    llm_client: Any = None,
) -> OracleBase:
    """Create an oracle from a dictionary definition.

    Used to instantiate oracles from scenario YAML ``oracle_definition``
    fields.  Supports nested composite oracles.

    Parameters
    ----------
    oracle_def : dict
        Oracle definition with ``"type"`` key and type-specific params.
    store : GMSExpertStore or None
        GMS store for geometric oracles.
    environment : EnvironmentSimulator or None
        Environment for side-effect oracle.
    llm_client : object or None
        Anthropic client for human-reviewed oracle.

    Returns
    -------
    OracleBase
    """
    otype = oracle_def.get("type", "exact_outcome")

    if otype == "exact_outcome":
        return ExactOutcomeOracle(store=store)

    elif otype == "acceptable_set":
        return AcceptableSetOracle(store=store)

    elif otype == "constraint":
        return ConstraintOracle(store=store)

    elif otype == "trace":
        return TraceOracle(store=store)

    elif otype == "side_effect":
        return SideEffectOracle(environment=environment, store=store)

    elif otype == "human_reviewed":
        return HumanReviewedOracle(
            store=store, llm_client=llm_client,
            model=oracle_def.get("model", "claude-sonnet-4-20250514"),
        )

    elif otype == "composite":
        sub_defs = oracle_def.get("oracles", [])
        threshold = oracle_def.get("threshold", 0.5)
        sub_oracles = []
        for sub in sub_defs:
            weight = sub.get("weight", 1.0)
            sub_oracle = make_oracle(sub, store, environment, llm_client)
            sub_oracles.append((sub_oracle, weight))
        return CompositeOracle(
            oracles=sub_oracles,
            threshold=threshold,
            fail_on_any_critical=oracle_def.get("fail_on_any_critical", True),
        )

    else:
        raise ValueError(f"Unknown oracle type: {otype!r}")
