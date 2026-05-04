# SPDX-License-Identifier: Apache-2.0
"""GMSH Parallel Execution — concurrent question evaluation.

Provides a thread-pool based executor that evaluates DOE questions in
parallel, significantly reducing wall-clock time for large benchmark
runs (e.g. 142 questions from ~33 min to ~8 min with 4 workers).

Each worker gets its own adapter instance to avoid shared-state issues.
Results are collected and merged into the DOEBenchmarkResult in the
original question order.

Usage::

    executor = ParallelExecutor(ParallelConfig(max_workers=4))
    results = executor.execute_batch(
        questions=doe_questions,
        evaluate_fn=lambda q, ctx: my_evaluator(q, ctx),
        context_fn=lambda dq: get_context(dq),
    )
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ParallelConfig:
    """Configuration for parallel execution.

    Attributes
    ----------
    max_workers : int
        Maximum number of concurrent worker threads.
    batch_size : int
        Number of questions to submit per batch (0 = all at once).
    timeout_per_question_s : float
        Timeout for each individual question evaluation in seconds
        (0 = no timeout).
    retry_on_error : bool
        Whether to retry a question on transient errors.
    max_retries : int
        Maximum retries per question when retry_on_error is True.
    retry_delay_s : float
        Delay between retries in seconds.
    """

    max_workers: int = 4
    batch_size: int = 0
    timeout_per_question_s: float = 120.0
    retry_on_error: bool = True
    max_retries: int = 2
    retry_delay_s: float = 1.0

    def to_dict(self) -> dict:
        return {
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "timeout_per_question_s": self.timeout_per_question_s,
            "retry_on_error": self.retry_on_error,
            "max_retries": self.max_retries,
            "retry_delay_s": self.retry_delay_s,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ParallelConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Question evaluation result (internal)
# ---------------------------------------------------------------------------

@dataclass
class _EvalResult:
    """Internal result from a single parallel evaluation."""

    index: int
    doe_qid: str = ""
    answer: str = ""
    raw_response: str = ""
    elapsed_s: float = 0.0
    error: str = ""
    retries: int = 0


# ---------------------------------------------------------------------------
# Parallel executor
# ---------------------------------------------------------------------------

class ParallelExecutor:
    """Execute DOE question evaluations in parallel.

    Parameters
    ----------
    config : ParallelConfig
        Parallelism configuration.
    on_progress : callable or None
        Optional callback ``(completed, total, result)`` called after
        each question completes.
    """

    def __init__(
        self,
        config: ParallelConfig | None = None,
        on_progress: Callable[[int, int, _EvalResult], None] | None = None,
    ):
        self.config = config or ParallelConfig()
        self.on_progress = on_progress

    def execute_batch(
        self,
        questions: list[tuple[int, str, str, str]],
        evaluate_fn: Callable[[str, str], str],
    ) -> list[_EvalResult]:
        """Execute a batch of question evaluations in parallel.

        Parameters
        ----------
        questions : list[tuple[int, str, str, str]]
            Each tuple is ``(index, doe_qid, question_text, context)``.
        evaluate_fn : callable
            ``evaluate_fn(question, context) -> answer_string``.
            Must be thread-safe (stateless or with thread-local state).

        Returns
        -------
        list[_EvalResult]
            Results in the original index order.
        """
        cfg = self.config
        total = len(questions)
        results: dict[int, _EvalResult] = {}
        completed = 0

        # Determine batching
        if cfg.batch_size > 0:
            batches = [
                questions[i:i + cfg.batch_size]
                for i in range(0, total, cfg.batch_size)
            ]
        else:
            batches = [questions]

        for batch in batches:
            with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
                future_to_item: dict[Future, tuple[int, str, str, str]] = {}

                for item in batch:
                    future = pool.submit(
                        self._evaluate_one, item, evaluate_fn,
                    )
                    future_to_item[future] = item

                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    idx = item[0]
                    try:
                        result = future.result(
                            timeout=cfg.timeout_per_question_s or None,
                        )
                    except Exception as exc:
                        result = _EvalResult(
                            index=idx,
                            doe_qid=item[1],
                            error=f"Future error: {type(exc).__name__}: {exc}",
                        )

                    results[idx] = result
                    completed += 1

                    if self.on_progress:
                        self.on_progress(completed, total, result)

        # Return in original order
        return [results[i] for i in sorted(results.keys())]

    def _evaluate_one(
        self,
        item: tuple[int, str, str, str],
        evaluate_fn: Callable[[str, str], str],
    ) -> _EvalResult:
        """Evaluate a single question with retry logic."""
        idx, doe_qid, question, context = item
        cfg = self.config
        max_attempts = (cfg.max_retries + 1) if cfg.retry_on_error else 1
        last_error = ""

        for attempt in range(max_attempts):
            t0 = time.time()
            try:
                raw = evaluate_fn(question, context)
                elapsed = time.time() - t0
                return _EvalResult(
                    index=idx,
                    doe_qid=doe_qid,
                    answer=raw,
                    raw_response=raw[:500] if raw else "",
                    elapsed_s=elapsed,
                    retries=attempt,
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < max_attempts - 1:
                    time.sleep(cfg.retry_delay_s)

        return _EvalResult(
            index=idx,
            doe_qid=doe_qid,
            error=last_error,
            retries=max_attempts - 1,
        )

    def execute_doe_batch(
        self,
        doe_questions: list,
        evaluate_fn: Callable[[str, str], str],
        context_fn: Callable[[Any], str],
    ) -> list[_EvalResult]:
        """Execute DOE questions with a context function.

        Convenience wrapper that extracts question text and context
        from DOEQuestion objects.

        Parameters
        ----------
        doe_questions : list[DOEQuestion]
            The DOE-expanded questions.
        evaluate_fn : callable
            ``evaluate_fn(question, context) -> answer_string``.
        context_fn : callable
            ``context_fn(dq) -> context_string`` for each DOEQuestion.

        Returns
        -------
        list[_EvalResult]
            Results in original order.
        """
        items = []
        for i, dq in enumerate(doe_questions):
            ctx = context_fn(dq)
            items.append((
                i,
                dq.doe_qid,
                dq.question.llm_prompt,
                ctx,
            ))
        return self.execute_batch(items, evaluate_fn)


# ---------------------------------------------------------------------------
# Utility: merge parallel results into DOEBenchmarkResult
# ---------------------------------------------------------------------------

def merge_parallel_results(
    eval_results: list[_EvalResult],
    doe_questions: list,
    score_fn: Callable[[str, str], Any],
) -> list[dict]:
    """Merge parallel evaluation results with scoring.

    Parameters
    ----------
    eval_results : list[_EvalResult]
        Parallel execution results (in order).
    doe_questions : list[DOEQuestion]
        Original DOE questions (same order).
    score_fn : callable
        ``score_fn(answer, ground_truth) -> ScoreResult`` with
        ``.correct`` and ``.detail`` attributes.

    Returns
    -------
    list[dict]
        Dicts suitable for constructing DOEQuestionResult objects.
    """
    merged = []
    for er, dq in zip(eval_results, doe_questions):
        q = dq.question
        if er.error:
            merged.append({
                "doe_qid": dq.doe_qid,
                "base_qid": dq.base_qid,
                "category": q.category,
                "question": q.natural_language,
                "ground_truth": str(q.ground_truth),
                "answer": "",
                "correct": False,
                "detail": f"Error: {er.error}",
                "factor_assignment": dq.factor_assignment,
                "raw_response": "",
            })
        else:
            score = score_fn(er.answer, q.ground_truth)
            merged.append({
                "doe_qid": dq.doe_qid,
                "base_qid": dq.base_qid,
                "category": q.category,
                "question": q.natural_language,
                "ground_truth": str(q.ground_truth),
                "answer": str(er.answer),
                "correct": score.correct,
                "detail": score.detail,
                "factor_assignment": dq.factor_assignment,
                "raw_response": er.raw_response,
            })
    return merged
