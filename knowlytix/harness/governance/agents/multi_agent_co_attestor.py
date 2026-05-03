# SPDX-License-Identifier: Apache-2.0
"""Three-state co-attestor for the multi-agent governance surface.

Pattern: structural binary (``can_execute``) + behavioral continuous
(per-agent tool-frequency PSI) -> three-state verdict
``pass`` / ``flag`` / ``block`` (Ch 13/15 lattice).

  * ``block`` -- structural permission check fails.
  * ``flag``  -- structural passes but per-agent tool-call PSI
    against the bundle's reference distribution exceeds the flag
    threshold.
  * ``pass``  -- both clear.

Distinct from Ch 11 OR-semantics because the behavioral attestor is
continuous (PSI value), not binary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from knowlytix.harness.governance.agents.multi_agent import (
    MultiAgentOrchestrator,
)
from knowlytix.harness.governance.agents.multi_agent_calibration import (
    PerAgentToolFrequency,
)


Verdict = Literal["pass", "flag", "block"]


@dataclass
class MultiAgentCoAttestorVerdict:
    """Three-state verdict on a single tool call.

    Attributes
    ----------
    verdict : str
        ``pass`` / ``flag`` / ``block``.
    structural_passed : bool
        True iff the role permission check allows the call.
    psi_value : float
        Population stability index of the recent tool-call window
        for this agent vs the reference baseline. ``nan`` when no
        baseline is available.
    blocking_reasons : list[str]
        Reasons from the structural attestor.
    flag_reasons : list[str]
        Reasons from the PSI attestor.
    """

    verdict: Verdict
    structural_passed: bool
    psi_value: float
    blocking_reasons: list[str]
    flag_reasons: list[str]
    role_name: str
    tool_name: str


@dataclass(frozen=True)
class MultiAgentCoAttestorConfig:
    """Per-agent PSI flag thresholds.

    A PSI value above ``flag_threshold`` (typical 0.10--0.25)
    raises a flag but does not block. The runtime can SLO-cap
    the flag rate so persistent flagging triggers recalibration.
    """

    flag_threshold: float = 0.10
    block_threshold: float = float("inf")  # never block on PSI alone
    min_window_calls: int = 5


def _psi_one_role(
    reference: dict[str, float],
    observed: dict[str, int],
    *,
    eps: float = 1e-9,
) -> float:
    """PSI between reference (normalized) and observed (counts)."""
    obs_total = max(1, sum(observed.values()))
    obs_norm = {k: v / obs_total for k, v in observed.items()}
    keys = set(reference) | set(obs_norm)
    psi = 0.0
    for k in keys:
        p = max(reference.get(k, 0.0), eps)
        q = max(obs_norm.get(k, 0.0), eps)
        psi += (q - p) * np.log(q / p)
    return float(psi)


class MultiAgentCoAttestor:
    """Three-state co-attestor combining role check + tool-frequency PSI.

    Construction:

      attestor = MultiAgentCoAttestor(
          orch=orch,
          reference_freq=PerAgentToolFrequency(...),
          config=MultiAgentCoAttestorConfig(flag_threshold=0.10),
      )
      v = attestor.evaluate(role_name, tool_name, args, recent_window)

    The ``recent_window`` is a per-role list of recent tool names
    that the caller maintains; the attestor computes PSI on that
    window vs the reference normalised distribution.
    """

    def __init__(
        self,
        orch: MultiAgentOrchestrator,
        reference_freq: PerAgentToolFrequency | None = None,
        *,
        config: MultiAgentCoAttestorConfig | None = None,
    ) -> None:
        self.orch = orch
        self.reference_freq = reference_freq
        self.config = config or MultiAgentCoAttestorConfig()
        if reference_freq is not None:
            self._reference_norm = reference_freq.normalized()
        else:
            self._reference_norm = {}

    def evaluate(
        self,
        role_name: str,
        tool_name: str,
        args: dict | None = None,
        recent_window: list[str] | None = None,
    ) -> MultiAgentCoAttestorVerdict:
        """Return the three-state verdict on a (role, tool) call."""
        agent = self.orch.agents.get(role_name)
        blocking_reasons: list[str] = []
        flag_reasons: list[str] = []

        if agent is None:
            return MultiAgentCoAttestorVerdict(
                verdict="block",
                structural_passed=False,
                psi_value=float("nan"),
                blocking_reasons=[f"unknown role '{role_name}'"],
                flag_reasons=[],
                role_name=role_name,
                tool_name=tool_name,
            )

        structural_passed = agent.can_execute(tool_name)
        if not structural_passed:
            blocking_reasons.append(
                f"{role_name} role denies '{tool_name}'"
            )

        # PSI on recent window
        psi_value = float("nan")
        ref = self._reference_norm.get(role_name)
        if (ref is not None and recent_window is not None
                and len(recent_window) >= self.config.min_window_calls):
            counts: dict[str, int] = {}
            for t in recent_window:
                counts[t] = counts.get(t, 0) + 1
            psi_value = _psi_one_role(ref, counts)
            if psi_value > self.config.flag_threshold:
                flag_reasons.append(
                    f"{role_name} tool-call PSI {psi_value:.4f} "
                    f"> flag {self.config.flag_threshold:.4f}"
                )

        if not structural_passed:
            verdict: Verdict = "block"
        elif flag_reasons:
            verdict = "flag"
        else:
            verdict = "pass"

        return MultiAgentCoAttestorVerdict(
            verdict=verdict,
            structural_passed=structural_passed,
            psi_value=psi_value,
            blocking_reasons=blocking_reasons,
            flag_reasons=flag_reasons,
            role_name=role_name,
            tool_name=tool_name,
        )
