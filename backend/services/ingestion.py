import re
import uuid
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import Settings
from backend.api.deps import get_vectorstore
from backend.services.metrics import metrics


def _safe_filename(name: str) -> str:
    base = Path(name).name
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", base)
    return (cleaned[:200] or "upload").strip("._") or "upload"


def _load_documents(path: Path, original_name: str) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .pdf, .txt, or .md.")

    docs = loader.load()
    for d in docs:
        d.metadata.setdefault("source", original_name)
    return docs


def ingest_bytes(
    data: bytes,
    original_filename: str,
    settings: Settings,
) -> tuple[str, int, str]:
    """
    Save raw bytes, chunk, embed, and upsert into Chroma.
    Returns (doc_id, chunks_added, stored_path).
    """
    if not data:
        raise ValueError("Empty file")

    doc_id = str(uuid.uuid4())
    safe = _safe_filename(original_filename)
    stored_name = f"{doc_id}_{safe}"
    dest = settings.upload_dir.resolve() / stored_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)

    documents = _load_documents(dest, Path(original_filename).name)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["chunk_index"] = i
        chunk.metadata["source"] = Path(original_filename).name

    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    vectorstore = get_vectorstore(settings)
    if chunks:
        vectorstore.add_documents(chunks, ids=ids)

    n_chunks = len(chunks)
    metrics.record_ingestion(n_chunks)
    return doc_id, n_chunks, str(dest)
