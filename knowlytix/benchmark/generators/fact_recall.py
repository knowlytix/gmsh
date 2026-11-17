# SPDX-License-Identifier: Apache-2.0
"""Fact Recall -- natural single-fact questions mined from graph triples.

Unlike :class:`ExactRecallGenerator`, which keys questions on the exact-numeric
register and phrases them by ENM key (``'Cloud Platform/Technology/Revenue'``),
this generator mines the asserted ``(head, relation, tail)`` triples and asks a
natural question -- ``"What is the revenue of cloud platform?"`` -- whose subject
is a real graph entity. That makes the questions bindable by an entity-grounded
retriever (the question names an entity, not an opaque key), so it is the
generator a RAG system under test is exercised with. Ground truth is the asserted
tail; the answer-recovery function reads it back from the graph (the functionality
check). Schema-agnostic: it phrases ``has_revenue`` as ``revenue`` and works on any
corpus's relations, with no per-schema key parsing.
"""

from knowlytix.benchmark.generators.base import (
    ANSWER_FORMATS,
    GeneratedQuestion,
    QuestionGenerator,
)

_NUMERIC = __import__("re").compile(r"^-?[\d,]*\.?\d+%?$")


def _phrase(relation: str) -> str:
    """The natural phrase for a relation: ``has_revenue`` -> ``revenue``."""
    base = relation[4:] if relation.startswith("has_") else relation
    return base.replace("_", " ").strip()


def _name(entity: str) -> str:
    """The natural surface for an entity id: ``cloud_platform`` -> ``cloud platform``.

    The subject of the question is a human-readable phrase, not the raw graph id, so the
    question reads naturally (``"... of suspicious activity reporting?"``) and a
    text retriever is not asked to match an underscored token. Mirrors ``_name`` in
    :mod:`relation_recall`; the tail ground truth is unchanged."""
    return str(entity).replace("_", " ").strip()


class FactRecallGenerator(QuestionGenerator):
    category = "fact_recall"
    #: relations that are structural / non-content, not worth a recall question.
    #: `has_feature` links a compound row entity back to its base feature, so a
    #: recall question over it just restates the entity id.
    exclude = ("in_section", "has_alias", "has_product", "has_value",
               "has_feature")

    def __init__(self, exclude: tuple[str, ...] | None = None):
        if exclude is not None:
            self.exclude = exclude

    def generate(self, graph) -> list[GeneratedQuestion]:
        questions: list[GeneratedQuestion] = []
        idx = 0
        seen: set[tuple[str, str]] = set()
        for h, r, t in graph.triples:
            # Case-insensitive: a relation derived from a column name carries
            # the source header's casing (`has_Feature`).
            if str(r).lower() in {e.lower() for e in self.exclude} \
                    or (h, r) in seen:
                continue
            # One question per (head, relation): a functional attribute has a
            # single answer; many-valued relations are left to set/aggregation
            # generators.
            if len(graph.query_triples(head=h, relation=r)) != 1:
                continue
            seen.add((h, r))
            phrase = _phrase(r)
            numeric = bool(_NUMERIC.match(str(t).strip()))

            def make_fn(hh, rr):
                def fn(g):
                    res = g.query_triples(head=hh, relation=rr)
                    return res[0][2] if res else None
                return fn

            atype = "float" if numeric else "str"
            q = GeneratedQuestion(
                qid=f"fact_recall_{idx:03d}",
                category=self.category,
                natural_language=f"What is the {phrase} of {_name(h)}?",
                graph_answer_fn=make_fn(h, r),
                ground_truth=t,   # the asserted tail (string); graph_answer_fn returns it
                llm_prompt=(f"State the {phrase} of {_name(h)} exactly as the report "
                            f"records it. {ANSWER_FORMATS[atype]}"),
                answer_type=atype,
                metadata={"head": h, "relation": r, "tail": t},
            )
            if self._validate(graph, q):
                questions.append(q)
                idx += 1
        return questions
