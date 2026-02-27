from pathlib import Path

from backend.config import get_settings
from backend.services.ingestion import ingest_bytes


def test_ingest_txt_creates_chunks_and_file(fake_vectorstore, isolated_settings):
    settings = get_settings()
    doc_id, n_chunks, stored = ingest_bytes(
        b"The quick brown fox jumps over the lazy dog. " * 20,
        "sample.txt",
        settings,
    )
    assert len(doc_id) == 36
    assert n_chunks >= 1
    assert Path(stored).is_file()


def test_ingest_empty_raises(fake_vectorstore):
    settings = get_settings()
    try:
        ingest_bytes(b"", "empty.txt", settings)
    except ValueError as e:
        assert "Empty" in str(e)
    else:
        raise AssertionError("expected ValueError")
