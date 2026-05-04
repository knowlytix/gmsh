# SPDX-License-Identifier: Apache-2.0
"""Rule-based co-attestor for the constrained-LLM gate.

The geometric gate (:class:`LogicVerifier`) checks whether a proposal
or claim is plausible *given the trained manifold*.  But manifolds
have blind spots: novel entities short-circuit to ``permit``,
mis-calibration can shift thresholds, and a malicious training set
could embed a contradicting triple as plausible.  A single gate is
not enough for high-risk deployment.

This module provides a second, rule-based attestor that runs
alongside :class:`LogicVerifier` with **OR-semantics on blocking**:
either gate firing means the proposal or claim is rejected.  The
co-attestor's rules are deterministic, written in plain Python, and
do not depend on the trained manifold --- so they catch failure
modes that geometric calibration cannot.

Three rule families are shipped by default:

* ``forbidden_relations`` --- relation names the LLM is not allowed
  to coin into the knowledge graph (e.g. anything not in the corpus
  schema).  Default: empty (permissive).  Set to a closed list when
  the agent's knowledge graph schema is fixed.

* ``numeric_bounds`` --- per-relation ``(low, high)`` ranges for
  ``ExtractedTriple.numeric_value``.  A claim like
  ``model has_accuracy 1.5`` is structurally invalid and must be
  blocked regardless of what the manifold says about it.

* ``forbidden_tool_args`` --- per-tool argument constraints (e.g.
  ``train_model`` may not be invoked with ``model_name='delete_all'``).
  Useful as an extra fence around the gateway's policy layer.

Operators MUST configure the rules to match the deployment.  The
default constructor returns an attestor with no rules set --- which
is equivalent to no co-attestor at all, by design: the chapter walks
through configuring each rule explicitly so the audit log shows
which rule fired on which input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowlytix.harness.governance.governed_llm import ExtractedTriple


@dataclass
class LLMCoAttestor:
    """Rule-based second attestor for ConstrainedLLM.

    All checks return ``{"allowed": bool, "reason": str}``.  When
    ``allowed=False``, the reason names the rule that fired so the
    audit log can attribute the block to a specific policy.
    """

    forbidden_relations: frozenset[str] = field(default_factory=frozenset)
    numeric_bounds: dict[str, tuple[float, float]] = field(
        default_factory=dict
    )
    forbidden_tool_args: dict[str, frozenset[str]] = field(
        default_factory=dict
    )

    def check_proposal(
        self,
        context: str,
        tool: str,
        args: dict,
    ) -> dict:
        """Return the rule decision for a proposed tool call.

        ``context`` is included in the signature for symmetry with
        :meth:`LogicVerifier.verify_action` --- some deployments will
        gate proposals on ``(context, tool)`` pairs.
        """
        forbidden = self.forbidden_tool_args.get(tool)
        if forbidden:
            for k, v in args.items():
                token = f"{k}={v}"
                if token in forbidden or str(v) in forbidden:
                    return {
                        "allowed": False,
                        "reason": (
                            f"forbidden_tool_args: {tool}({token})"
                        ),
                    }
        return {"allowed": True, "reason": "no_rule_fired"}

    def check_claim(self, triple: "ExtractedTriple") -> dict:
        """Return the rule decision for an extracted claim.

        Order: forbidden relation first, then numeric bound on the
        triple's ``numeric_value``.  The first rule to fire wins,
        and its name is returned so the audit log can attribute the
        block.
        """
        if triple.relation in self.forbidden_relations:
            return {
                "allowed": False,
                "reason": (
                    f"forbidden_relation: '{triple.relation}'"
                ),
            }

        bounds = self.numeric_bounds.get(triple.relation)
        if bounds is not None and triple.numeric_value is not None:
            low, high = bounds
            v = triple.numeric_value
            if not (low <= v <= high):
                return {
                    "allowed": False,
                    "reason": (
                        f"numeric_bound: {triple.relation}={v} "
                        f"outside [{low}, {high}]"
                    ),
                }

        return {"allowed": True, "reason": "no_rule_fired"}


def default_ds_co_attestor() -> LLMCoAttestor:
    """Default rules for the data-science workflow used in Chapter 11.

    These are the minimal rules a banking-style model-development
    agent would carry: structural ranges on the standard metrics
    (accuracy, F1, precision, recall, AUC --- all bounded in
    ``[0, 1]``), forbidden-relation list left empty so the agent can
    coin new relation names within the schema, and
    ``forbidden_tool_args`` blocking the obvious destructive args.
    """
    return LLMCoAttestor(
        forbidden_relations=frozenset(),
        numeric_bounds={
            "has_accuracy":   (0.0, 1.0),
            "has_f1_score":   (0.0, 1.0),
            "has_f1":         (0.0, 1.0),
            "has_precision":  (0.0, 1.0),
            "has_recall":     (0.0, 1.0),
            "has_auc":        (0.0, 1.0),
            "has_roc_auc":    (0.0, 1.0),
            "has_importance": (0.0, 1.0),
        },
        forbidden_tool_args={
            "train_model": frozenset({
                "model_name=delete_all",
                "model_name=__rm__",
            }),
            "preprocess_data": frozenset({
                "drop_all=True",
            }),
        },
    )
