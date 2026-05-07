# SPDX-License-Identifier: Apache-2.0
"""Session state for the governed sklearn agent.

Mirrors agents/ds_agent/tools.py SessionState but wraps sklearn
instead of MoDeVa.  Every value stored here is a plain Python object
(no live sklearn estimators leak into serialized results).

Built in Chapter 3 of "How to Build Policy-Governed AI Agents."
"""

from dataclasses import dataclass, field


@dataclass
class SessionState:
    """Mutable state persisting across tool calls within a session."""

    datasets: dict = field(default_factory=dict)
    # name -> {X_train, X_test, y_train, y_test,
    #          feature_names, target_name, task_type, df}

    models: dict = field(default_factory=dict)
    # name -> {estimator, model_type, params, dataset}

    results: dict = field(default_factory=dict)
    # name -> {metric: value, ...}

    preprocessors: dict = field(default_factory=dict)
    # name -> fitted sklearn Pipeline

    predictions: dict = field(default_factory=dict)
    # name -> {y_pred, y_prob (optional)}
