"""In-process metrics for query latency and ingestion/retrieval counters."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock


_LATENCY_CAPACITY = 200


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Nearest-rank percentile on a non-empty pre-sorted list."""
    if not sorted_vals:
        raise ValueError("empty sample")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    # Nearest-rank: index = ceil(p/100 * n) - 1
    rank = max(1, int((p / 100.0) * len(sorted_vals) + 0.999999))
    return sorted_vals[min(rank, len(sorted_vals)) - 1]


@dataclass
class MetricsStore:
    _lock: Lock = field(default_factory=Lock, repr=False)
    _latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=_LATENCY_CAPACITY))
    queries_total: int = 0
    ingestions_total: int = 0
    chunks_added_total: int = 0
    retrieval_returned_total: int = 0
    retrieval_kept_total: int = 0

    def record_ingestion(self, chunks_added: int) -> None:
        with self._lock:
            self.ingestions_total += 1
            self.chunks_added_total += max(0, int(chunks_added))

    def record_retrieval(self, returned: int, kept: int) -> None:
        with self._lock:
            self.retrieval_returned_total += max(0, int(returned))
            self.retrieval_kept_total += max(0, int(kept))

    def record_query(self, latency_ms: float) -> None:
        with self._lock:
            self.queries_total += 1
            self._latencies_ms.append(float(latency_ms))

    def reset(self) -> None:
        with self._lock:
            self._latencies_ms.clear()
            self.queries_total = 0
            self.ingestions_total = 0
            self.chunks_added_total = 0
            self.retrieval_returned_total = 0
            self.retrieval_kept_total = 0

    def snapshot(self) -> dict:
        with self._lock:
            samples = list(self._latencies_ms)
            out: dict = {
                "queries_total": self.queries_total,
                "ingestions_total": self.ingestions_total,
                "chunks_added_total": self.chunks_added_total,
                "retrieval_returned_total": self.retrieval_returned_total,
                "retrieval_kept_total": self.retrieval_kept_total,
                "latency_samples": len(samples),
                "query_latency_p50_ms": None,
                "query_latency_p95_ms": None,
            }
            if len(samples) >= 2:
                ordered = sorted(samples)
                out["query_latency_p50_ms"] = round(_percentile(ordered, 50), 3)
                out["query_latency_p95_ms"] = round(_percentile(ordered, 95), 3)
            return out


metrics = MetricsStore()
