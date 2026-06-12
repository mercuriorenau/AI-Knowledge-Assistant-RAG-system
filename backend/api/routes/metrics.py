from fastapi import APIRouter

from backend.models.schemas import MetricsResponse
from backend.services.metrics import metrics

router = APIRouter(tags=["metrics"])


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    snap = metrics.snapshot()
    return MetricsResponse(**snap)
