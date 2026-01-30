from fastapi import APIRouter

from backend.config import get_settings
from backend.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    s = get_settings()
    return HealthResponse(
        status="ok",
        openai_configured=bool(s.openai_api_key),
    )
