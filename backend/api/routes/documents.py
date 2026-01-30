import re
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.config import get_settings
from backend.models.schemas import DocumentListItem, DocumentUploadResponse
from backend.services.ingestion import ingest_bytes

router = APIRouter(tags=["documents"])

_UUID_RE = re.compile(
    r"^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})_(.+)$",
    re.IGNORECASE,
)


@router.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Set OPENAI_API_KEY in the environment.",
        )
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload a .pdf, .txt, or .md file.",
        )

    data = await file.read()
    try:
        doc_id, chunks_added, stored_path = ingest_bytes(data, file.filename, settings)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e!s}") from e

    return DocumentUploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        chunks_added=chunks_added,
        stored_path=stored_path,
    )


@router.get("/documents", response_model=list[DocumentListItem])
def list_documents() -> list[DocumentListItem]:
    settings = get_settings()
    upload_dir = settings.upload_dir.resolve()
    if not upload_dir.exists():
        return []

    items: list[DocumentListItem] = []
    for p in sorted(upload_dir.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        m = _UUID_RE.match(p.name)
        if not m:
            continue
        doc_id, filename = m.group(1), m.group(2)
        items.append(
            DocumentListItem(doc_id=doc_id, filename=filename, stored_path=str(p))
        )
    return items
