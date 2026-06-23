from langchain_core.documents import Document

from backend.config import Settings
from backend.api.deps import get_vectorstore
from backend.models.schemas import SourceItem
from backend.services.metrics import metrics


def retrieve_chunks(
    question: str,
    settings: Settings,
    k: int | None = None,
) -> tuple[list[tuple[Document, float]], str]:
    """
    Return (list of (document, relevance_score), assembled_context_string).
    Chroma returns distances from similarity_search_with_score; we map them to (0, 1]
    via 1 / (1 + |distance|) so MIN_RELEVANCE_SCORE works across embedding backends.
    """
    k = k or settings.retrieval_k
    vectorstore = get_vectorstore(settings)
    raw = vectorstore.similarity_search_with_score(question, k=k)
    pairs = [(d, 1.0 / (1.0 + abs(float(dist)))) for d, dist in raw]

    filtered = [(d, s) for d, s in pairs if s >= settings.min_relevance_score]
    metrics.record_retrieval(returned=len(pairs), kept=len(filtered))

    blocks = []
    for doc, _score in filtered:
        src = doc.metadata.get("source", "unknown")
        blocks.append(f"[Source: {src}]\n{doc.page_content}")
    context = "\n\n".join(blocks)
    return filtered, context


def pairs_to_sources(pairs: list[tuple[Document, float]], snippet_len: int = 320) -> list[SourceItem]:
    out: list[SourceItem] = []
    for doc, score in pairs:
        text = doc.page_content.strip()
        snippet = text if len(text) <= snippet_len else text[:snippet_len].rsplit(" ", 1)[0] + " …"
        out.append(
            SourceItem(
                doc_id=str(doc.metadata.get("doc_id", "")),
                source=str(doc.metadata.get("source", "unknown")),
                chunk_index=doc.metadata.get("chunk_index"),
                snippet=snippet,
                chunk_text=text,
                relevance_score=float(score),
            )
        )
    return out
