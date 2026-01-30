from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from backend.config import Settings, get_settings


@lru_cache(maxsize=32)
def _vectorstore_for(
    persist_s: str,
    collection: str,
    embedding_model: str,
    openai_api_key: str,
) -> Chroma:
    Path(persist_s).mkdir(parents=True, exist_ok=True)
    emb_kwargs: dict = {"model": embedding_model}
    if openai_api_key:
        emb_kwargs["api_key"] = openai_api_key
    embeddings = OpenAIEmbeddings(**emb_kwargs)
    return Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=persist_s,
    )


def get_vectorstore(settings: Settings | None = None) -> Chroma:
    s = settings or get_settings()
    persist = str(s.chroma_persist_dir.resolve())
    return _vectorstore_for(
        persist,
        s.chroma_collection_name,
        s.embedding_model,
        s.openai_api_key,
    )


def clear_vectorstore_cache() -> None:
    """Test helper: reset cached Chroma clients after changing persist dir or credentials."""
    _vectorstore_for.cache_clear()
