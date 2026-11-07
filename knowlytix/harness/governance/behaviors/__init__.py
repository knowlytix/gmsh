# SPDX-License-Identifier: Apache-2.0
"""Behavioral contracts: FSM-based structural enforcement for governed agents.

Public API:

* :class:`BehavioralContract`, :class:`ProcessEvaluator`,
  :class:`SequenceValidation`, :class:`ContractState` --- core FSM
  contract abstraction and validation (Ch.13).
* :func:`advisory_contract`, :func:`recommendation_contract`,
  :func:`action_taking_contract` --- three pre-defined risk-tier
  contracts.
* :class:`ContractAdversarialProbe`, :class:`ContractAdversarialReport`,
  :func:`build_contract_adversarial_cohort`,
  :func:`sweep_contract_adversarial`, :func:`cohort_hash` ---
  systematic adversarial cohort that exercises every block path.
* :class:`HolonomyCoAttestor`, :class:`TwoAttestorContractVerdict`,
  :func:`canonical_compliant_workflows` --- holonomy-based
  co-attestor with three-state verdict (pass / flag / block).
* :class:`BehavioralContractBundle`, :class:`ContractFingerprint`,
  :class:`ContractOperatingPoint`,
  :func:`build_behavioral_contract_bundle`,
  :func:`bind_contract_from_bundle` --- HMAC-signed frozen artifact.
* :class:`ContractReplayResult`, :class:`ContractReplayRegressionError`,
  :func:`replay_contract`, :func:`assert_no_contract_regression` ---
  byte-identical replay regression battery.
"""

from knowlytix.harness.governance.behaviors.contract_adversarial import (
    ContractAdversarialProbe,
    ContractAdversarialReport,
    build_contract_adversarial_cohort,
    cohort_hash,
    sweep_contract_adversarial,
)
from knowlytix.harness.governance.behaviors.contract_bundle import (
    CONTRACT_BUNDLE_SCHEMA_VERSION,
    BehavioralContractBundle,
    ContractFingerprint,
    ContractOperatingPoint,
    bind_contract_from_bundle,
    build_behavioral_contract_bundle,
    compute_contract_fingerprint,
)
from knowlytix.harness.governance.behaviors.contract_co_attestor import (
    HolonomyCoAttestor,
    TwoAttestorContractVerdict,
    canonical_compliant_workflows,
)
from knowlytix.harness.governance.behaviors.contract_replay import (
    ContractReplayRegressionError,
    ContractReplayResult,
    assert_no_contract_regression,
    replay_contract,
)
from knowlytix.harness.governance.behaviors.contracts import (
    ALL_TOOLS,
    TOOL_STATE_ADVANCE,
    BehavioralContract,
    ContractState,
    ProcessEvaluator,
    SequenceValidation,
    TextGuardrail,
    action_taking_contract,
    advisory_contract,
    recommendation_contract,
)

__all__ = [
    "ALL_TOOLS",
    "BehavioralContract",
    "BehavioralContractBundle",
    "CONTRACT_BUNDLE_SCHEMA_VERSION",
    "ContractAdversarialProbe",
    "ContractAdversarialReport",
    "ContractFingerprint",
    "ContractOperatingPoint",
    "ContractReplayRegressionError",
    "ContractReplayResult",
    "ContractState",
    "HolonomyCoAttestor",
    "ProcessEvaluator",
    "SequenceValidation",
    "TextGuardrail",
    "TOOL_STATE_ADVANCE",
    "TwoAttestorContractVerdict",
    "action_taking_contract",
    "advisory_contract",
    "assert_no_contract_regression",
    "bind_contract_from_bundle",
    "build_behavioral_contract_bundle",
    "build_contract_adversarial_cohort",
    "canonical_compliant_workflows",
    "cohort_hash",
    "compute_contract_fingerprint",
    "recommendation_contract",
    "replay_contract",
    "sweep_contract_adversarial",
]
