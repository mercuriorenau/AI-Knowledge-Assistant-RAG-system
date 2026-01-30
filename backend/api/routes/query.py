from fastapi import APIRouter, HTTPException

from backend.config import get_settings
from backend.models.schemas import QueryRequest, QueryResponse
from backend.services.rag import answer_question

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest) -> QueryResponse:
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Set OPENAI_API_KEY in the environment.",
        )
    try:
        return answer_question(body.question.strip(), settings, k=body.k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e!s}") from e
