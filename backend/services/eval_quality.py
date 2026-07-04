"""Precision and recall helpers for retrieval evaluation."""

from __future__ import annotations


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for item in top if item in relevant)
    return hits / len(top)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    hits = sum(1 for item in relevant if item in top)
    return hits / len(relevant)


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
