from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""

    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"

    chroma_persist_dir: Path = Path("./chroma_data")
    upload_dir: Path = Path("./backend/storage/uploads")
    chroma_collection_name: str = "kb_default"

    chunk_size: int = 900
    chunk_overlap: int = 120

    retrieval_k: int = 8
    min_relevance_score: float = 0.25
    llm_temperature: float = 0.2


def get_settings() -> Settings:
    return Settings()
