import pytest
from fastapi.testclient import TestClient

from backend.api.deps import clear_vectorstore_cache
from backend.config import get_settings
from backend.main import app
from backend.services.ingestion import ingest_bytes


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_vectorstore_cache()
    yield
    clear_vectorstore_cache()


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "openai_configured" in body


def test_query_without_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    clear_vectorstore_cache()
    client = TestClient(app)
    r = client.post("/query", json={"question": "Hello?"})
    assert r.status_code == 503


def test_upload_without_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    clear_vectorstore_cache()
    client = TestClient(app)
    files = {"file": ("x.txt", b"data", "text/plain")}
    r = client.post("/documents", files=files)
    assert r.status_code == 503


def test_upload_and_query_end_to_end(fake_vectorstore, monkeypatch):
    monkeypatch.setattr(
        "backend.services.rag.ChatOpenAI",
        lambda **kwargs: _FakeLLM(),
    )

    ingest_bytes(
        b"Acme Corp was founded in 1999. Its HQ is in Austin.",
        "acme.txt",
        get_settings(),
    )

    client = TestClient(app)
    up = client.post(
        "/documents",
        files={"file": ("extra.txt", b"Secondary doc.", "text/plain")},
    )
    assert up.status_code == 200
    assert up.json()["chunks_added"] >= 1

    r = client.post("/query", json={"question": "When was Acme founded?"})
    assert r.status_code == 200
    data = r.json()
    assert "1999" in data["answer"]
    assert data["context_empty"] is False
    assert len(data["sources"]) >= 1


class _FakeLLM:
    def invoke(self, messages):
        class R:
            content = (
                "According to the documents, Acme Corp was founded in 1999. "
                "(Source: acme.txt)"
            )

        return R()
