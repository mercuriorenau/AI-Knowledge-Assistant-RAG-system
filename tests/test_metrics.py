from fastapi.testclient import TestClient

from backend.main import app
from backend.services.metrics import MetricsStore, _percentile, metrics


def test_percentile_nearest_rank():
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert _percentile(vals, 50) == 30.0
    assert _percentile(vals, 95) == 50.0


def test_snapshot_omits_percentiles_until_two_samples():
    store = MetricsStore()
    store.record_query(12.5)
    snap = store.snapshot()
    assert snap["queries_total"] == 1
    assert snap["latency_samples"] == 1
    assert snap["query_latency_p50_ms"] is None
    assert snap["query_latency_p95_ms"] is None

    store.record_query(20.0)
    snap2 = store.snapshot()
    assert snap2["latency_samples"] == 2
    assert snap2["query_latency_p50_ms"] is not None
    assert snap2["query_latency_p95_ms"] is not None


def test_metrics_endpoint_after_ingest(fake_vectorstore, isolated_settings):
    metrics.reset()
    client = TestClient(app)
    from backend.config import get_settings
    from backend.services.ingestion import ingest_bytes

    ingest_bytes(b"hello world " * 5, "m.txt", get_settings())
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["ingestions_total"] == 1
    assert body["chunks_added_total"] >= 1


def test_retrieval_counters_survive_relevance_floor(fake_vectorstore, isolated_settings, monkeypatch):
    metrics.reset()
    monkeypatch.setenv("MIN_RELEVANCE_SCORE", "0.99")
    from backend.config import get_settings
    from backend.services.ingestion import ingest_bytes
    from backend.services.retrieval import retrieve_chunks

    settings = get_settings()
    ingest_bytes(b"Acme founded in 1999 in Austin Texas.", "acme.txt", settings)
    retrieve_chunks("When was Acme founded?", settings, k=4)
    snap = metrics.snapshot()
    assert snap["retrieval_returned_total"] >= 1
    assert snap["retrieval_kept_total"] <= snap["retrieval_returned_total"]
