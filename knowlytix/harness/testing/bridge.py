# SPDX-License-Identifier: Apache-2.0
"""Bridge between eval generators (GMS-based) and graphDOE question format.

The eval generators produce ``knowlytix.benchmark.eval.generators.base.GeneratedQuestion``
objects that carry a ``gms_answer_fn(store)`` callable.  The graphDOE rephraser
expects ``knowlytix.benchmark.generators.base.GeneratedQuestion`` which carries
``graph_answer_fn(graph)``.

This module adapts one to the other so the DOE design matrix expansion and
question rephrasing pipeline works unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from knowlytix.benchmark.eval.generators.base import GeneratedQuestion as EvalQuestion
from knowlytix.benchmark.generators.base import GeneratedQuestion as FSBQuestion

if TYPE_CHECKING:
    from knowlytix.knowledge.store import GMSExpertStore

    from .scenarios import Scenario


# -- Generator name → class mapping (for selective loading) -----------------

_GENERATOR_MAP = {
    # Classic eval generators (plain graph queries)
    "exact_recall": "ExactRecallGenerator",
    "threshold": "ThresholdGenerator",
    "cross_reference": "CrossReferenceGenerator",
    "contradiction": "ContradictionGenerator",
    "multi_hop": "MultiHopGenerator",
    "counting": "CountingGenerator",
    # GMS geometric generators (manifold-native)
    "plausibility": "PlausibilityGenerator",
    "tension": "TensionGenerator",
    "holonomy": "HolonomyGenerator",
    "link_prediction": "LinkPredictionGenerator",
    "multi_hop_transport": "MultiHopTransportGenerator",
    "phase_comparison": "PhaseComparisonGenerator",
    "conditional": "ConditionalGenerator",
    "procedural": "ProceduralGenerator",
    # GMSH extended generators
    "exact_retrieval": "ExactRetrievalGenerator",
    "negative": "NegativeGenerator",
    "workflow_compliance": "WorkflowComplianceGenerator",
    # GMSH taxonomy generators
    "boundary": "BoundaryGenerator",
    "adversarial": "AdversarialGenerator",
    "robustness": "RobustnessGenerator",
    "recovery_test": "RecoveryTestGenerator",
    "governance": "GovernanceTestGenerator",
    # GMS-native enhancement generators
    "stateful_memory": "StatefulAgentGenerator",
    "hallucination": "HallucinationGenerator",
    "calibration": "CalibrationGenerator",
    "multi_agent": "MultiAgentGenerator",
    "cross_document": "CrossDocumentGenerator",
    "disambiguation": "DisambiguationGenerator",
    "paraphrase_invariance": "InvarianceGenerator",
}


def adapt_question(eq: EvalQuestion, store: "GMSExpertStore") -> FSBQuestion:
    """Convert an eval GeneratedQuestion to a finstructbench GeneratedQuestion.

    The key mapping: ``gms_answer_fn(store)`` becomes
    ``graph_answer_fn(graph)`` via a closure that ignores the graph argument
    and calls the original function with the captured store.
    """
    return FSBQuestion(
        qid=eq.qid,
        category=eq.category,
        natural_language=eq.natural_language,
        graph_answer_fn=lambda _graph, _s=store, _fn=eq.gms_answer_fn: _fn(_s),
        ground_truth=eq.ground_truth,
        llm_prompt=eq.llm_prompt,
        answer_type=eq.answer_type,
        metadata=eq.metadata,
    )


def generate_base_questions(
    store: "GMSExpertStore",
    generator_names: list[str] | None = None,
    max_per_category: int = 10,
    seed: int = 42,
) -> list[FSBQuestion]:
    """Run eval generators against a GMS store and return DOE-compatible questions.

    Parameters
    ----------
    store : GMSExpertStore
        The ingested document store (ground truth).
    generator_names : list[str] or None
        Which generators to use.  ``None`` or empty means all.
    max_per_category : int
        Maximum questions per generator category.
    seed : int
        Deterministic sampling seed.

    Returns
    -------
    list[FSBQuestion]
        Questions ready for DOE expansion via ``QuestionRephraser.expand()``.
    """
    # Geometric, extended, and taxonomy generator names
    _GEOMETRIC_NAMES = {
        "plausibility", "tension", "holonomy",
        "link_prediction", "multi_hop_transport", "phase_comparison",
        "conditional", "procedural",
    }
    _EXTENDED_NAMES = {
        "exact_retrieval", "negative", "workflow_compliance",
    }
    _TAXONOMY_NAMES = {
        "boundary", "adversarial", "robustness",
        "recovery_test", "governance",
    }
    _ENHANCEMENT_NAMES = {
        "stateful_memory", "hallucination", "calibration",
        "multi_agent", "cross_document", "disambiguation",
        "paraphrase_invariance",
    }

    if generator_names:
        from knowlytix.benchmark.eval import generators as gen_mod
        from knowlytix.harness.testing.calibration import CalibrationGenerator
        from knowlytix.harness.testing.cross_document import CrossDocumentGenerator
        from knowlytix.harness.testing.disambiguation import DisambiguationGenerator
        from knowlytix.harness.testing.generators import (
            geometric_generators as _geo_gens,
        )
        from knowlytix.harness.testing.generators_taxonomy import (
            taxonomy_generators as _tax_gens,
        )
        from knowlytix.harness.testing.hallucination import HallucinationGenerator
        from knowlytix.harness.testing.invariance import InvarianceGenerator
        from knowlytix.harness.testing.multi_agent import MultiAgentGenerator
        from knowlytix.harness.testing.stateful import StatefulAgentGenerator
        from knowlytix.harness.testing.test_generators import (
            extended_generators as _ext_gens,
        )

        # Build lookup for geometric, extended, and taxonomy generators
        geo_lookup = {type(g).category if hasattr(type(g), 'category')
                      else g.category: g for g in _geo_gens()}
        ext_lookup = {type(g).category if hasattr(type(g), 'category')
                      else g.category: g for g in _ext_gens()}
        tax_lookup = {type(g).category if hasattr(type(g), 'category')
                      else g.category: g for g in _tax_gens()}
        enh_lookup = {
            "stateful_memory": StatefulAgentGenerator(),
            "hallucination": HallucinationGenerator(),
            "calibration": CalibrationGenerator(),
            "multi_agent": MultiAgentGenerator(),
            "cross_document": CrossDocumentGenerator(),
            "disambiguation": DisambiguationGenerator(),
            "paraphrase_invariance": InvarianceGenerator(),
        }

        gens = []
        for name in generator_names:
            if name in _GEOMETRIC_NAMES:
                gen = geo_lookup.get(name)
                if gen is None:
                    raise ValueError(f"Unknown geometric generator '{name}'")
                gens.append(gen)
            elif name in _EXTENDED_NAMES:
                gen = ext_lookup.get(name)
                if gen is None:
                    raise ValueError(f"Unknown extended generator '{name}'")
                gens.append(gen)
            elif name in _TAXONOMY_NAMES:
                gen = tax_lookup.get(name)
                if gen is None:
                    raise ValueError(f"Unknown taxonomy generator '{name}'")
                gens.append(gen)
            elif name in _ENHANCEMENT_NAMES:
                gen = enh_lookup.get(name)
                if gen is None:
                    raise ValueError(f"Unknown enhancement generator '{name}'")
                gens.append(gen)
            else:
                cls_name = _GENERATOR_MAP.get(name)
                if cls_name is None:
                    raise ValueError(
                        f"Unknown generator '{name}'. "
                        f"Available: {list(_GENERATOR_MAP)}"
                    )
                cls = getattr(gen_mod, cls_name)
                gens.append(cls())
    else:
        # Default: geometric (8) + taxonomy (5) = 13 non-redundant generators.
        # Classic and extended generators are superseded by their geometric
        # counterparts (ExactRecall→LinkPrediction, Contradiction→Tension,
        # MultiHop→MultiHopTransport, CrossReference→Conditional,
        # WorkflowCompliance→Procedural).  They remain available by name
        # for backward compatibility but are no longer in the default set.
        # Enhancement generators are opt-in via config flags.
        from knowlytix.harness.testing.generators import geometric_generators
        from knowlytix.harness.testing.generators_taxonomy import taxonomy_generators
        gens = geometric_generators() + taxonomy_generators()

    all_questions: list[FSBQuestion] = []
    for gen in gens:
        candidates = gen.generate(store)
        sampled = gen.sample(candidates, max_per_category, seed)
        # Validate against store
        valid = [q for q in sampled if gen._validate(store, q)]
        for eq in valid:
            all_questions.append(adapt_question(eq, store))

    print(f"  Base questions: {len(all_questions)} "
          f"({len(gens)} generators, max {max_per_category}/cat)")
    return all_questions


def scenario_to_fsb_question(
    scenario: "Scenario",
    store: "GMSExpertStore | None" = None,
) -> FSBQuestion | None:
    """Convert a Scenario into a DOE-compatible FSBQuestion.

    If the scenario has ``question_text`` and ``ground_truth``, build the
    question directly.  If it has ``generator_name``, delegate to the
    named generator (requires ``store``).

    Parameters
    ----------
    scenario : Scenario
        The scenario to convert.
    store : GMSExpertStore or None
        Required when using ``generator_name``.

    Returns
    -------
    FSBQuestion or None
        The converted question, or None if conversion fails.
    """

    # Manually authored scenario — build directly
    if scenario.question_text and scenario.ground_truth:
        metadata = dict(scenario.metadata)
        # Inject behavioral requirements into metadata for extended evaluators
        if scenario.required_tools:
            metadata["required_tools"] = scenario.required_tools
        if scenario.forbidden_tools:
            metadata["forbidden_tools"] = scenario.forbidden_tools
        if scenario.expected_tool_sequence:
            metadata["expected_tool_sequence"] = scenario.expected_tool_sequence
        if scenario.max_tool_calls:
            metadata["max_tool_calls"] = scenario.max_tool_calls
        if scenario.expected_recovery:
            metadata["expected_recovery"] = scenario.expected_recovery
        metadata["scenario_id"] = scenario.scenario_id
        metadata["risk_tier"] = scenario.risk_tier

        return FSBQuestion(
            qid=scenario.scenario_id,
            category=f"scenario_{scenario.risk_tier}",
            natural_language=scenario.question_text,
            graph_answer_fn=lambda _graph, _gt=scenario.ground_truth: _gt,
            ground_truth=scenario.ground_truth,
            llm_prompt=scenario.question_text,
            answer_type=scenario.answer_type or "str",
            metadata=metadata,
        )

    # Generator-based scenario — delegate to named generator
    if scenario.generator_name and store is not None:
        gen_name = scenario.generator_name
        if gen_name not in _GENERATOR_MAP:
            return None

        from knowlytix.benchmark.eval import generators as gen_mod
        cls_name = _GENERATOR_MAP[gen_name]
        cls = getattr(gen_mod, cls_name, None)
        if cls is None:
            return None

        gen = cls(**scenario.generator_params) if scenario.generator_params else cls()
        candidates = gen.generate(store)
        valid = [q for q in candidates if gen._validate(store, q)]
        if not valid:
            return None

        # Take the first valid question and adapt it
        eq = valid[0]
        fsb_q = adapt_question(eq, store)
        # Override metadata with scenario info
        fsb_q.metadata = {**fsb_q.metadata, "scenario_id": scenario.scenario_id}
        return fsb_q

    return None
