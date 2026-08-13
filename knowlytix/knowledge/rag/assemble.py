# SPDX-License-Identifier: Apache-2.0
"""Answer assembly: synthesize a grounded answer from retrieved facts + spans.

The synthesis LLM (user's choice) sees the GMS-retrieved facts and their source
spans and is instructed to answer ONLY from them — so the answer is grounded in
verified structure, not the model's parametric memory.
"""

from __future__ import annotations

from knowlytix.knowledge.llm_backend import LLMBackend
from knowlytix.knowledge.rag.retrieve import RetrievedFact

__all__ = ["Assembler", "ABSTAIN_SENTINEL"]

_SYSTEM = (
    "You are given GMS-verified FACTS, each written as 'FACT: head | relation | "
    "tail', that have ALREADY been selected as the relevant grounded knowledge for "
    "the question. Your only job is to VERBALIZE these facts -- not to judge "
    "whether they fully answer the question, and not to add anything not present. "
    "State EVERY provided FACT in full as a short declarative sentence that keeps "
    "the subject (head), the attribute (relation) and the value (tail) together, "
    "so all three appear in your answer. For 'FACT: overdraft | has_fee_amount | "
    "35.0' say 'The overdraft fee is $35.0.'; for 'FACT: manager | has_max_reversal "
    "| 500.0' say 'A manager can reverse up to 500.0.'; for 'FACT: disputes | "
    "has_provisional_credit | issued' say 'Disputes have provisional credit "
    "issued.'. When several facts are given (e.g. a multi-hop chain), state ALL of "
    "them. Keep every value exactly as given; never reply with a bare value, a "
    "relation name without its value, or a yes/no alone. "
    "NEVER say 'cannot answer', 'the facts do not provide ...', or any refusal "
    "while at least one FACT line is present -- simply state the facts. Abstain "
    "ONLY when there are literally no FACT lines. "
    "Do NOT restate the question and do NOT narrate or show reasoning steps, even "
    "if the question says to think step by step."
)

# Select-and-answer prompt for the operator-native ``head_facts`` retrieval mode
# (RagConfig.retrieve_mode="head_facts"). There, retrieval returns ALL of the
# bound head's admissible facts -- not a single pre-selected one -- so synthesis
# must (1) pick the fact(s) that answer the question, (2) chain facts for a
# multi-hop question, and (3) abstain when none of them answers it. Relevance
# moves here, grounded on the retrieved facts, instead of a separate gate.
ABSTAIN_SENTINEL = "I cannot answer this from the available grounded evidence."
_SELECT_SYSTEM = (
    "You are given GMS-verified FACTS about the question's subject, each written "
    "as 'FACT: head | relation | tail'. These are ALL the grounded facts known "
    "about that subject. Answer the question using ONLY the fact(s) that actually "
    "address what it asks, as one short declarative sentence that keeps the head, "
    "the attribute and the value together (for 'FACT: retail | has_headcount | "
    "520.0' and 'how many people work in retail' say 'Retail headcount is 520.0.'). "
    "You MAY combine facts that chain together to answer a multi-step question "
    "(e.g. 'FACT: logistics | has_division | operations' and 'FACT: operations | "
    "has_head | sam reyes' answer 'who heads the division that contains logistics' "
    "-> 'Sam Reyes heads operations, the division containing logistics.'). Keep "
    "every value exactly as given. When the answering fact's value is a STANCE or "
    "permission -- forbidden, prohibited, permitted, allowed, required, optional, "
    "issued, denied -- STATE THAT STANCE DIRECTLY and do NOT convert it into an "
    "affirmative yes/no: for 'FACT: pii_handling | has_unencrypted_channel_pii | "
    "forbidden' and 'Can my SSN be sent over email?' answer 'Sending a Social "
    "Security number over an unencrypted channel is forbidden.' -- never 'Yes, it "
    "can be sent.'. If NONE of the facts addresses what the question asks, reply "
    "EXACTLY: " + repr(ABSTAIN_SENTINEL) + ". Never use a fact that does not answer "
    "the question. Do not restate the question or show reasoning."
)


class Assembler:
    """Synthesize a grounded answer from retrieved facts.

    ``mode="verbalize"`` (default) states every provided FACT and never refuses
    while a fact is present -- for the ``query_triple`` path where the relevant
    fact was already selected upstream. ``mode="select"`` picks the answering
    fact(s) from all of a head's retrieved facts and abstains when none answer --
    for the operator-native ``head_facts`` path (relevance moves into synthesis).
    """

    def __init__(self, llm: LLMBackend, *, max_tokens: int = 1024,
                 mode: str = "verbalize"):
        self.llm = llm
        self.max_tokens = max_tokens
        self.system = _SELECT_SYSTEM if mode == "select" else _SYSTEM

    def assemble(self, question: str, facts: list[RetrievedFact]) -> str:
        return self.llm.call(system=self.system,
                            user=self._context(question, facts),
                            max_tokens=self.max_tokens)

    def assemble_passages(self, question: str, passages: list[str]) -> str:
        """Synthesize from raw text passages (dense fallback — unverified)."""
        body = "\n\n".join(f"PASSAGE:\n{p}" for p in passages) or "(none)"
        user = (f"<passages>\n{body}\n</passages>\n\nQuestion: {question}")
        return self.llm.call(system=_SYSTEM, user=user, max_tokens=self.max_tokens)

    @staticmethod
    def _context(question: str, facts: list[RetrievedFact]) -> str:
        lines: list[str] = []
        for f in facts:
            lines.append(f"FACT: {f.head} | {f.relation} | {f.tail}")
            if f.raw:
                loc = f" ({f.location})" if f.location else ""
                lines.append(f"SOURCE{loc}: {f.raw}")
        evidence = "\n".join(lines) if lines else "(no facts retrieved)"
        return (f"<evidence>\n{evidence}\n</evidence>\n\n"
                f"Question: {question}")
