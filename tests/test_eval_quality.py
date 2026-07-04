import json
from pathlib import Path

from backend.services.eval_quality import mean, precision_at_k, recall_at_k


def test_precision_recall_helpers():
    retrieved = ["acme.txt", "helios.txt", "northwind.txt"]
    relevant = {"acme.txt"}
    assert precision_at_k(retrieved, relevant, k=1) == 1.0
    assert precision_at_k(retrieved, relevant, k=3) == 1 / 3
    assert recall_at_k(retrieved, relevant, k=1) == 1.0
    assert recall_at_k(["helios.txt"], relevant, k=1) == 0.0
    assert mean([1.0, 0.0]) == 0.5


def test_qa_fixture_shape():
    qa_path = Path(__file__).resolve().parents[1] / "evals" / "qa.json"
    cases = json.loads(qa_path.read_text(encoding="utf-8"))
    assert len(cases) >= 3
    for case in cases:
        assert case["question"]
        assert case["relevant_sources"]
