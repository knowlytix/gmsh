# SPDX-License-Identifier: Apache-2.0
"""GMSH Streaming Output Testing — GMS-native chunk consistency analysis.

Tests streaming LLM outputs for consistency using three GMS primitives:

- **Per-chunk plausibility**: ``score_triple`` on each chunk measures
  intermediate answer plausibility on the manifold.
- **Stream self-contradiction**: ``tension_energy`` between first-chunk
  entities and final-answer entities detects mid-stream contradiction.
- **Chunk path coherence**: ``check_holonomy`` across the logical chain
  of chunk entities measures whether the stream's reasoning path is
  geometrically consistent.

Usage::

    adapter = StreamingAdapter(client, model, store)
    trace = adapter.evaluate("What is X?", context)
    print(trace.first_token_ms, trace.stream_tension, trace.chunk_holonomy)

    oracle = StreamConsistencyOracle(store)
    verdict = oracle.evaluate_stream(trace)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anthropic
    from knowlytix.knowledge.store import GMSExpertStore


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class StreamChunk:
    """A single chunk from a streaming response.

    Attributes
    ----------
    index : int
        Chunk sequence number (0-based).
    text : str
        Raw text of this chunk.
    cumulative_text : str
        All text up to and including this chunk.
    timestamp_ms : float
        Milliseconds since stream start.
    plausibility_geodesic : float or None
        ``score_triple`` geodesic for entities in this chunk.
    """

    index: int = 0
    text: str = ""
    cumulative_text: str = ""
    timestamp_ms: float = 0.0
    plausibility_geodesic: float | None = None


@dataclass
class StreamingTrace:
    """Extended trace for streaming output with GMS consistency metrics.

    Attributes
    ----------
    raw_output : str
        Complete final output.
    chunks : list[StreamChunk]
        Per-chunk data.
    first_token_ms : float
        Time to first token in milliseconds.
    total_latency_ms : float
        Total stream time in milliseconds.
    stream_tension : float or None
        Tension energy between first-chunk and final entities.
        High tension = the stream contradicted its early output.
    chunk_holonomy : float or None
        Holonomy defect across the chunk sequence.
        Zero = coherent reasoning path.
    mean_chunk_plausibility : float
        Mean geodesic across all chunk plausibility scores.
    n_chunks : int
        Total number of chunks.
    error : str
        Error message if streaming failed.
    """

    raw_output: str = ""
    chunks: list[StreamChunk] = field(default_factory=list)
    first_token_ms: float = 0.0
    total_latency_ms: float = 0.0
    stream_tension: float | None = None
    chunk_holonomy: float | None = None
    mean_chunk_plausibility: float = 0.0
    n_chunks: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "n_chunks": self.n_chunks,
            "first_token_ms": round(self.first_token_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "stream_tension": (
                round(self.stream_tension, 4)
                if self.stream_tension is not None else None
            ),
            "chunk_holonomy": (
                round(self.chunk_holonomy, 4)
                if self.chunk_holonomy is not None else None
            ),
            "mean_chunk_plausibility": round(self.mean_chunk_plausibility, 4),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Streaming adapter
# ---------------------------------------------------------------------------

class StreamingAdapter:
    """Adapter that captures streaming output with per-chunk GMS scoring.

    Uses the Anthropic streaming API to capture each chunk, then scores
    intermediate plausibility via ``score_triple``.

    Parameters
    ----------
    client : anthropic.Anthropic
        Anthropic API client.
    model : str
        Model to stream from.
    store : GMSExpertStore or None
        GMS store for per-chunk plausibility scoring.
    """

    def __init__(
        self,
        client: "anthropic.Anthropic",
        model: str,
        store: "GMSExpertStore | None" = None,
    ):
        self.client = client
        self.model = model
        self.store = store

    def evaluate(self, question: str, context: str) -> StreamingTrace:
        """Stream a response and capture per-chunk metrics.

        Parameters
        ----------
        question : str
            The question to ask.
        context : str
            Context/report text.

        Returns
        -------
        StreamingTrace
            Full streaming trace with GMS metrics.
        """
        chunks: list[StreamChunk] = []
        cumulative = ""
        t0 = time.time()
        first_token_ms = 0.0

        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": (
                        f"<report>\n{context}\n</report>\n\n{question}"
                    ),
                }],
            ) as stream:
                for i, event in enumerate(stream.text_stream):
                    now_ms = (time.time() - t0) * 1000
                    if i == 0:
                        first_token_ms = now_ms

                    cumulative += event
                    chunk = StreamChunk(
                        index=i,
                        text=event,
                        cumulative_text=cumulative,
                        timestamp_ms=now_ms,
                    )

                    # Score chunk plausibility if store available
                    if self.store is not None:
                        chunk.plausibility_geodesic = self._score_chunk(
                            event, cumulative,
                        )

                    chunks.append(chunk)

            total_ms = (time.time() - t0) * 1000

            # Compute stream-level GMS metrics
            stream_tension = self._compute_stream_tension(chunks)
            chunk_holonomy = self._compute_chunk_holonomy(chunks)
            mean_plaus = self._mean_plausibility(chunks)

            return StreamingTrace(
                raw_output=cumulative,
                chunks=chunks,
                first_token_ms=first_token_ms,
                total_latency_ms=total_ms,
                stream_tension=stream_tension,
                chunk_holonomy=chunk_holonomy,
                mean_chunk_plausibility=mean_plaus,
                n_chunks=len(chunks),
            )

        except Exception as exc:
            total_ms = (time.time() - t0) * 1000
            return StreamingTrace(
                raw_output=cumulative,
                chunks=chunks,
                first_token_ms=first_token_ms,
                total_latency_ms=total_ms,
                n_chunks=len(chunks),
                error=str(exc),
            )

    def _score_chunk(self, chunk_text: str, cumulative: str) -> float | None:
        """Score a chunk's plausibility using score_triple."""
        if self.store is None:
            return None
        # Use the chunk's first word as a rough entity proxy
        words = chunk_text.strip().split()
        if not words:
            return None
        entity = words[0]
        try:
            matched = self.store.fuzzy_match_entity(entity)
            if matched:
                score = self.store.score_triple(
                    matched, "mentioned_in", "stream_chunk",
                )
                return float(score) if score is not None else None
        except Exception:
            pass
        return None

    def _compute_stream_tension(
        self, chunks: list[StreamChunk],
    ) -> float | None:
        """Tension between first-chunk and final entities."""
        if self.store is None or len(chunks) < 2:
            return None
        first_text = chunks[0].text.strip()
        final_text = chunks[-1].cumulative_text.strip()
        first_words = first_text.split()[:3]
        final_words = final_text.split()[-3:]

        if not first_words or not final_words:
            return None

        try:
            first_ent = self.store.fuzzy_match_entity(first_words[0])
            final_ent = self.store.fuzzy_match_entity(final_words[-1])
            if first_ent and final_ent and first_ent != final_ent:
                return self.store.tension_energy(first_ent, final_ent)
        except Exception:
            pass
        return None

    def _compute_chunk_holonomy(
        self, chunks: list[StreamChunk],
    ) -> float | None:
        """Holonomy defect across chunk entity path."""
        if self.store is None or len(chunks) < 3:
            return None
        # Extract entity path from chunks
        path_relations = []
        for chunk in chunks[:5]:  # first 5 chunks
            words = chunk.text.strip().split()
            if words:
                matched = self.store.fuzzy_match_entity(words[0])
                if matched and matched not in path_relations:
                    path_relations.append(matched)

        if len(path_relations) < 2:
            return None

        # Check holonomy of the entity path
        try:
            return self.store.check_holonomy(
                path_relations[:-1], path_relations[-1],
            )
        except Exception:
            return None

    def _mean_plausibility(self, chunks: list[StreamChunk]) -> float:
        """Mean plausibility geodesic across scored chunks."""
        scored = [
            c.plausibility_geodesic for c in chunks
            if c.plausibility_geodesic is not None
        ]
        return sum(scored) / len(scored) if scored else 0.0


# ---------------------------------------------------------------------------
# Stream consistency oracle
# ---------------------------------------------------------------------------

class StreamConsistencyOracle:
    """Evaluate streaming output consistency using GMS metrics.

    Parameters
    ----------
    store : GMSExpertStore
        GMS store for scoring.
    tension_threshold : float
        Maximum stream tension for consistency (default 0.5).
    holonomy_threshold : float
        Maximum holonomy defect for coherence (default 0.5).
    """

    oracle_type = "stream_consistency"

    def __init__(
        self,
        store: "GMSExpertStore",
        tension_threshold: float = 0.5,
        holonomy_threshold: float = 0.5,
    ):
        self.store = store
        self.tension_threshold = tension_threshold
        self.holonomy_threshold = holonomy_threshold

    def evaluate_stream(self, trace: StreamingTrace) -> dict:
        """Evaluate a streaming trace for consistency.

        Parameters
        ----------
        trace : StreamingTrace
            The streaming trace to evaluate.

        Returns
        -------
        dict
            Verdict with pass/fail, tension, holonomy, and detail.
        """
        tension_ok = (
            trace.stream_tension is None
            or trace.stream_tension <= self.tension_threshold
        )
        holonomy_ok = (
            trace.chunk_holonomy is None
            or trace.chunk_holonomy <= self.holonomy_threshold
        )

        passed = tension_ok and holonomy_ok

        detail_parts = []
        if not tension_ok:
            detail_parts.append(
                f"Stream tension {trace.stream_tension:.4f} > "
                f"threshold {self.tension_threshold}"
            )
        if not holonomy_ok:
            detail_parts.append(
                f"Chunk holonomy {trace.chunk_holonomy:.4f} > "
                f"threshold {self.holonomy_threshold}"
            )

        return {
            "oracle_type": self.oracle_type,
            "passed": passed,
            "stream_tension": trace.stream_tension,
            "chunk_holonomy": trace.chunk_holonomy,
            "mean_chunk_plausibility": trace.mean_chunk_plausibility,
            "first_token_ms": trace.first_token_ms,
            "n_chunks": trace.n_chunks,
            "detail": "; ".join(detail_parts) if detail_parts else "Consistent",
        }
