# SPDX-License-Identifier: Apache-2.0
"""
DOE Benchmark Results — stores scored results with factor annotations.
"""

import json
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class DOEQuestionResult:
    """Result for a single DOE-expanded question.

    ``correct`` is the strict exact-string match against ``ground_truth``.
    ``path_valid`` is True iff every hop the system traversed
    (``path_chain``) is a real edge in the source store — useful complement
    for multi-valued relations where exact-string can mark a question wrong
    even when the retrieval landed on a different but graph-truthful fork.
    """

    doe_qid: str
    base_qid: str
    category: str
    question: str
    ground_truth: str
    answer: str
    correct: bool
    detail: str
    factor_assignment: dict = field(default_factory=dict)
    raw_response: str = ""
    # Graph-truth metrics (optional; populated by harnesses that can read
    # the underlying retrieval chain). Default values keep older results
    # files loadable.
    path_chain: list = field(default_factory=list)  # list[(h, r, t)]
    path_valid: bool | None = None
    path_valid_detail: str = ""


@dataclass
class DOEBenchmarkResult:
    """Full result of a DOE benchmark run."""

    instance_name: str = ""
    total_questions: int = 0
    total_correct: int = 0
    total_path_valid: int = 0       # number of questions with valid retrieval chain
    n_base_questions: int = 0
    n_design_runs: int = 0
    factors: list[str] = field(default_factory=list)
    model: str = ""
    timestamp: str = ""
    context_mode: str = "full"
    graph_stats: dict = field(default_factory=dict)
    questions: list[DOEQuestionResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if self.total_questions == 0:
            return 0.0
        return self.total_correct / self.total_questions

    @property
    def path_valid_rate(self) -> float:
        """Fraction of questions whose retrieved chain consists entirely of
        real edges in the source store. Complement to ``accuracy`` for
        multi-valued relations."""
        if self.total_questions == 0:
            return 0.0
        return self.total_path_valid / self.total_questions

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a DataFrame for analysis.

        Each row is one DOE question with columns:
        doe_qid, base_qid, category, correct, + one column per factor.
        """
        rows = []
        for qr in self.questions:
            row = {
                "doe_qid": qr.doe_qid,
                "base_qid": qr.base_qid,
                "category": qr.category,
                "correct": int(qr.correct),
                "ground_truth": qr.ground_truth,
                "answer": qr.answer,
                "detail": qr.detail,
                "path_valid": (None if qr.path_valid is None
                               else int(qr.path_valid)),
                "path_valid_detail": qr.path_valid_detail,
            }
            row.update(qr.factor_assignment)
            rows.append(row)
        return pd.DataFrame(rows)

    def save(self, path: str):
        """Save results to JSON."""
        data = {
            "instance": self.instance_name,
            "total_questions": self.total_questions,
            "total_correct": self.total_correct,
            "total_path_valid": self.total_path_valid,
            "accuracy": self.accuracy,
            "path_valid_rate": self.path_valid_rate,
            "n_base_questions": self.n_base_questions,
            "n_design_runs": self.n_design_runs,
            "factors": self.factors,
            "model": self.model,
            "timestamp": self.timestamp,
            "context_mode": self.context_mode,
            "graph_stats": self.graph_stats,
            "questions": [
                {
                    "doe_qid": q.doe_qid,
                    "base_qid": q.base_qid,
                    "category": q.category,
                    "question": q.question,
                    "ground_truth": q.ground_truth,
                    "answer": q.answer,
                    "correct": q.correct,
                    "detail": q.detail,
                    "factors": q.factor_assignment,
                    "path_chain": q.path_chain,
                    "path_valid": q.path_valid,
                    "path_valid_detail": q.path_valid_detail,
                }
                for q in self.questions
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DOEBenchmarkResult":
        """Load results from JSON."""
        with open(path) as f:
            data = json.load(f)
        result = cls(
            instance_name=data["instance"],
            total_questions=data["total_questions"],
            total_correct=data["total_correct"],
            total_path_valid=data.get("total_path_valid", 0),
            n_base_questions=data["n_base_questions"],
            n_design_runs=data["n_design_runs"],
            factors=data["factors"],
            model=data.get("model", ""),
            timestamp=data.get("timestamp", ""),
            context_mode=data.get("context_mode", "full"),
            graph_stats=data.get("graph_stats", {}),
        )
        for q in data["questions"]:
            result.questions.append(
                DOEQuestionResult(
                    doe_qid=q["doe_qid"],
                    base_qid=q["base_qid"],
                    category=q["category"],
                    question=q["question"],
                    ground_truth=q["ground_truth"],
                    answer=q["answer"],
                    correct=q["correct"],
                    detail=q.get("detail", ""),
                    factor_assignment=q.get("factors", {}),
                    path_chain=q.get("path_chain", []),
                    path_valid=q.get("path_valid"),
                    path_valid_detail=q.get("path_valid_detail", ""),
                )
            )
        return result
