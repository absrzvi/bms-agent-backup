"""FastAPI application exposing document ingestion and semantic search APIs."""
from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from .processor_wrapper import DocumentProcessorWrapper

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".xlsx", ".xls"}
DEFAULT_UPLOAD_ROOT = Path(os.getenv("BMS_UPLOAD_PATH", Path.home() / "persistent" / "bms_data" / "uploads"))
MAX_UPLOAD_BYTES = int(os.getenv("BMS_UPLOAD_MAX_BYTES", str(100 * 1024 * 1024)))  # 100 MB

API_KEY = os.getenv("BMS_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

processor = DocumentProcessorWrapper()

_metrics: Dict[str, Optional[float]] = {
    "documents_processed": 0,
    "search_requests": 0,
    "last_processing_started": None,
    "last_processing_duration": None,
    "last_search_latency": None,
}

app = FastAPI(
    title="BMS Agent API",
    version="1.0.0",
    description="Railway documentation retrieval APIs",
    docs_url="/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("BMS_ALLOWED_ORIGINS", "http://localhost:5678,http://localhost:8080").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_api_key(api_key: Optional[str] = Depends(api_key_header)) -> str:
    if API_KEY:
        if api_key != API_KEY:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")
        return API_KEY
    # If no API key configured, accept anonymous access for MVP usage
    return api_key or "anonymous"


class SemanticSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1024)
    limit: int = Field(5, ge=1, le=50)


@app.get("/", tags=["Info"])
async def root() -> Dict[str, str]:
    return {
        "service": "BMS Agent API",
        "version": app.version,
        "description": app.description,
    }


@app.get("/health", tags=["Health"])
async def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/health/detailed", tags=["Health"])
async def detailed_health(_: str = Depends(verify_api_key)) -> Dict[str, object]:
    try:
        collections = processor.qdrant.get_collections()
        qdrant_ok = True
        collection_names = [c.name for c in collections.collections]
        collection_exists = processor.collection_name in collection_names
    except Exception as exc:  # pragma: no cover - requires Qdrant
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Qdrant unavailable: {exc}") from exc

    return {
        "api": "healthy",
        "qdrant": qdrant_ok,
        "collection_exists": collection_exists,
        "collection_name": processor.collection_name,
        "embedding_model": processor.embedding_model,
        "metrics": _metrics,
    }


@app.get("/metrics/uplink", tags=["Metrics"])
async def metrics(_: str = Depends(verify_api_key)) -> Dict[str, Optional[float]]:
    return _metrics


@app.post("/api/v1/search/semantic", tags=["Search"])
async def semantic_search(payload: SemanticSearchRequest, _: str = Depends(verify_api_key)) -> Dict[str, object]:
    start = datetime.utcnow()
    results = await processor.search_documents(payload.query, limit=payload.limit)
    _metrics["search_requests"] = (_metrics["search_requests"] or 0) + 1
    _metrics["last_search_latency"] = (datetime.utcnow() - start).total_seconds()
    return results


@app.post("/api/v1/documents/upload", tags=["Documents"])
async def upload_document(file: UploadFile = File(...), _: str = Depends(verify_api_key)) -> JSONResponse:
    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{extension}'. Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    upload_root = DEFAULT_UPLOAD_ROOT
    upload_root.mkdir(parents=True, exist_ok=True)
    destination = upload_root / file.filename

    size = 0
    start_time = datetime.utcnow()
    async with aiofile_writer(destination) as writer:
        while True:
            chunk = await file.read(1 << 20)  # 1 MB
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                await file.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds limit of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
                )
            await writer(chunk)

    result = await processor.process_document(destination)
    duration = (datetime.utcnow() - start_time).total_seconds()
    _metrics["documents_processed"] = (_metrics["documents_processed"] or 0) + 1
    _metrics["last_processing_started"] = start_time.isoformat()
    _metrics["last_processing_duration"] = duration

    response_payload = {
        "status": "success" if result.get("processing_success", False) else "warning",
        "filename": file.filename,
        "bytes_stored": size,
        "processing": result,
    }
    return JSONResponse(status_code=status.HTTP_200_OK, content=response_payload)


class aiofile_writer:
    """Async context manager that exposes an awaitable writer callback."""

    def __init__(self, path: Path):
        self.path = path
        self._file = None

    async def __aenter__(self):
        loop = asyncio.get_running_loop()
        self._file = await loop.run_in_executor(None, self.path.open, "wb")

        async def _writer(data: bytes) -> None:
            if not self._file:
                raise RuntimeError("file handle not initialized")
            await loop.run_in_executor(None, self._file.write, data)

        self._writer = _writer
        return self._writer

    async def __aexit__(self, exc_type, exc, tb):
        if self._file:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._file.flush)
            await loop.run_in_executor(None, self._file.close)
        self._writer = None
        return False
