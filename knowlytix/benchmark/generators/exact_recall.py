# SPDX-License-Identifier: Apache-2.0
"""Exact Recall — mines ENM for scalar value questions."""

from knowlytix.benchmark.generators.base import (
    ANSWER_FORMATS,
    GeneratedQuestion,
    QuestionGenerator,
)


class ExactRecallGenerator(QuestionGenerator):
    """Exact recall (base ``exact_recall``): one question per ENM entry, read
    back its stored value.

    GMS mapping: ground_truth=lookup_enm · requires=[enm] · answer_type=float.
    Gates when: the graph has no ENM entries.
    """
    category = "exact_recall"

    def generate(self, graph) -> list[GeneratedQuestion]:
        """Generate one exact-value recall question per validated ENM entry in ``graph``.

        Args:
            graph: the document graph whose ENM entries seed the questions.

        Returns:
            The validated :class:`GeneratedQuestion` objects, each keyed to an ENM
            ``(type, id)`` with a graph lookup as its answer function.
        """
        questions = []
        idx = 0

        for key, entry in list(graph.enm.items()):
            cat, eid = key.type, key.id

            def make_fn(c, e):
                def fn(g):
                    return g.lookup(c, e)
                return fn

            q = GeneratedQuestion(
                qid=f"exact_recall_{idx:03d}",
                category=self.category,
                natural_language=f"What is the exact value of '{eid}' (category: {cat})?",
                graph_answer_fn=make_fn(cat, eid),
                ground_truth=entry.value,
                llm_prompt=(
                    f"Find the exact numeric value for '{eid}' in the "
                    f"'{cat}' section of the report. "
                    f"Copy the number exactly as shown with ALL decimal places. "
                    f"{ANSWER_FORMATS['float']}"
                ),
                answer_type="float",
                metadata={"enm_type": cat, "enm_id": eid},
            )

            if self._validate(graph, q):
                questions.append(q)
                idx += 1

        return questions
