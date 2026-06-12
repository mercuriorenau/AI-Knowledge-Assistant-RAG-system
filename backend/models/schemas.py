from pydantic import BaseModel, Field


class SourceItem(BaseModel):
    doc_id: str
    source: str
    chunk_index: int | None = None
    snippet: str = Field(description="Short excerpt from the retrieved chunk")
    chunk_text: str = Field(description="Full retrieved chunk text")
    relevance_score: float


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    k: int | None = Field(default=None, ge=1, le=50)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    context_empty: bool = False


class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_added: int
    stored_path: str


class DocumentListItem(BaseModel):
    doc_id: str
    filename: str
    stored_path: str


class HealthResponse(BaseModel):
    status: str
    openai_configured: bool


class MetricsResponse(BaseModel):
    queries_total: int
    ingestions_total: int
    chunks_added_total: int
    retrieval_returned_total: int
    retrieval_kept_total: int
    latency_samples: int
    query_latency_p50_ms: float | None = None
    query_latency_p95_ms: float | None = None
