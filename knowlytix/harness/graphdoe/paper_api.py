# SPDX-License-Identifier: Apache-2.0
"""
Paper-style designed-experiment API.

Mirrors the FinStructBench case study from `graphdoe/papers/paper_jasa.tex`:
each design row picks one base question AND a configuration of contextual
factors, so the total LLM-call budget equals ``n_runs`` (not
``n_questions × n_runs``).  The base questions are treated as a single
categorical factor named ``question`` with cardinality equal to the
number of supplied questions.

Quick start::

    import graphdoe

    # 30 graph-verified base questions extracted from a GMS store, plus
    # 10 contextual factors and 300 design rows -> 300 LLM calls.
    design = graphdoe.design(
        questions=base_question_strings,            # length 30
        factor_names=["clarity", "noise", "role",
                      "answer_format", "reasoning_cue",
                      "context_scope", "entity_aliasing",
                      "instruction_conflict", "complexity",
                      "style"],                     # 10 contextual factors
        n_runs=300,
        method="sobol+refine",
    )
    print(design.quality())     # phi_p, min_dist, ...
    df = design.matrix          # 300 rows; columns = ["question",
                                #   "clarity", ..., "style"]

The returned object is a ``DesignMatrix`` with ``question`` listed first
in ``factors`` and present as a column of ``matrix`` / ``df``.  Iterate
its rows, look up the base question for each row, render the prompt with
the row's contextual factor levels, send to the LLM, score against the
graph-verified ground truth, and pass the resulting DataFrame to
``DOEAnalyzer`` for factor-effect inference.
"""

from __future__ import annotations

from .design_matrix import DesignMatrix
from .factor_loader import FactorLoader


__all__ = ["design"]


def design(
    questions: list,
    *,
    factor_group: str | None = None,
    factor_names: list[str] | None = None,
    custom_factors: list[dict] | None = None,
    n_runs: int = 200,
    method: str = "sobol+refine",
    seed: int = 42,
    verbose: bool = False,
    catalog_path: str | None = None,
) -> DesignMatrix:
    """Build a paper-style design with ``question`` as a categorical factor.

    Parameters
    ----------
    questions
        Base questions.  Items may be plain strings or any object with a
        ``natural_language`` attribute (e.g. a ``GeneratedQuestion``);
        the function extracts the string used as the categorical level
        label and as the design column value.
    factor_group
        Name of a factor group from the catalog (e.g. ``"comprehensive"``
        for 20 factors, ``"core"`` for 10).  Combined with
        ``factor_names`` and ``custom_factors`` if more than one is
        supplied.
    factor_names
        Explicit list of factor names from the catalog.
    custom_factors
        Raw factor dictionaries (must include ``name``, ``type``, and
        ``categories``/``cardinality`` for categorical, ``min``/``max``
        for continuous).
    n_runs
        Number of design rows = number of LLM calls when each row is
        evaluated once.
    method
        ``"sobol"``, ``"sobol+refine"``, ``"lhs"``, or ``"grid"``.  Pass
        ``"sobol+refine"`` for the GPU phi_p-optimized design.
    seed
        Random seed for reproducibility.
    verbose
        Print optimization progress for ``method="sobol+refine"``.
    catalog_path
        Override the bundled factor catalog.

    Returns
    -------
    DesignMatrix
        With ``question`` as the first factor and one row per LLM call.
        ``.matrix`` contains the categorical-level DataFrame; ``.quality()``
        returns space-filling metrics.
    """
    if not questions:
        raise ValueError("questions must be a non-empty list")

    # Normalize questions to strings (also accepts objects with .natural_language)
    question_strings = []
    for q in questions:
        if isinstance(q, str):
            question_strings.append(q)
        elif hasattr(q, "natural_language"):
            question_strings.append(q.natural_language)
        else:
            raise TypeError(
                f"questions[{len(question_strings)}] must be str or have "
                f"a .natural_language attribute; got {type(q).__name__}"
            )

    # Deduplicate while preserving order; categorical levels must be unique
    seen = set()
    unique_questions = []
    for q in question_strings:
        if q not in seen:
            seen.add(q)
            unique_questions.append(q)
    if len(unique_questions) != len(question_strings):
        raise ValueError(
            f"questions contains duplicates "
            f"({len(question_strings) - len(unique_questions)} dropped); "
            "all categorical levels must be unique"
        )

    question_factor = {
        "name": "question",
        "type": "categorical",
        "cardinality": len(unique_questions),
        "categories": unique_questions,
    }

    # Resolve contextual factors from the catalog
    loader = FactorLoader(catalog_path)
    config: dict = {}
    if factor_group:
        config["use_group"] = factor_group
    if factor_names:
        config["use_factors"] = factor_names
    if custom_factors:
        config["custom_factors"] = custom_factors
    contextual = loader.load_factors(config) if config else []

    # Avoid clobbering the question factor if the user asked for one named "question"
    contextual = [f for f in contextual if f["name"] != "question"]

    all_factors = [question_factor] + contextual
    loader.validate_factors(all_factors)

    dm = DesignMatrix(
        all_factors,
        method=method,
        n_runs=n_runs,
        seed=seed,
        verbose=verbose,
    )
    dm.generate()
    return dm
