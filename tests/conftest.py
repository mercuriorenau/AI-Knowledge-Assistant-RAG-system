import pytest
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma

from backend.api.deps import clear_vectorstore_cache


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("MIN_RELEVANCE_SCORE", "0.0")
    clear_vectorstore_cache()
    yield tmp_path
    clear_vectorstore_cache()


@pytest.fixture
def fake_vectorstore(isolated_settings, monkeypatch):
    tmp_path = isolated_settings
    emb = FakeEmbeddings(size=135)
    vs = Chroma(
        collection_name="kb_default",
        embedding_function=emb,
        persist_directory=str(tmp_path / "chroma"),
    )

    def _vs(settings=None):
        return vs

    monkeypatch.setattr("backend.services.ingestion.get_vectorstore", _vs)
    monkeypatch.setattr("backend.services.retrieval.get_vectorstore", _vs)
    return vs
