# SPDX-License-Identifier: Apache-2.0
"""
DOE Benchmark Results — stores scored results with factor annotations.
"""

import json
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class DOEQuestionResult:
    """Result for a single DOE-expanded question."""

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


@dataclass
class DOEBenchmarkResult:
    """Full result of a DOE benchmark run."""

    instance_name: str = ""
    total_questions: int = 0
    total_correct: int = 0
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
            "accuracy": self.accuracy,
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
                )
            )
        return result
