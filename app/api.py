"""
potbot — FastAPI Adapter (Minimal API Bridge)

Thin HTTP layer that exposes the existing RAGPipeline and IngestionPipeline
services to the React frontend. No business logic lives here — every handler
delegates directly to the same Python services Streamlit called internally.

Endpoints
---------
GET  /api/health              — liveness probe
GET  /api/status              — knowledge base stats (ES doc/chunk count + online status)
POST /api/chat                — RAG query → answer + sources + metadata
POST /api/feedback            — save thumbs-up/down feedback
POST /api/ingest/files        — upload files and run ingestion pipeline
POST /api/ingest/folder       — run ingestion pipeline on a server-side folder path
GET  /api/supported-extensions — list of file extensions the loader supports
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional

# Ensure root is in PYTHONPATH when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
from app.database import PostgresDatabaseRepository
from domain.models import FeedbackRecord
from ingestion.loaders import CompositeDocumentLoader
from ingestion.pipeline import IngestionPipeline
from rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="potbot API",
    description="Internal Intelligence Platform — REST API adapter for the React frontend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restricted in production via reverse proxy
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Service singletons (lazily initialised, cached for the lifetime of the process)
# ---------------------------------------------------------------------------

_db_repo: Optional[PostgresDatabaseRepository] = None
_rag_pipeline: Optional[RAGPipeline] = None


def _get_db_repo() -> PostgresDatabaseRepository:
    global _db_repo
    if _db_repo is None:
        _db_repo = PostgresDatabaseRepository()
        try:
            _db_repo.init_db()
        except Exception as e:
            logger.warning("DB init deferred: %s", e)
    return _db_repo


def _get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline(repository=_get_db_repo())
    return _rag_pipeline


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question")
    retrieval_method: str = Field("hybrid", description="hybrid | vector | text")
    use_reranking: bool = Field(True)
    use_query_rewriting: bool = Field(True)
    prompt_style: str = Field("detailed", description="detailed | concise | structured")
    llm_provider: str = Field("groq", description="groq | ollama")
    llm_model: str = Field("", description="Model identifier; empty = use default")


class SourceItem(BaseModel):
    file_name: str
    page_number: Optional[int]
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    conversation_id: Optional[int]
    response_time_ms: int
    total_tokens: int
    model: str
    retrieval_method: str
    prompt_style: str


class FeedbackRequest(BaseModel):
    conversation_id: int
    sentiment: str = Field(..., description="positive | negative")
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    feedback_id: int
    message: str


class StatusResponse(BaseModel):
    online: bool
    status_text: str
    unique_file_count: int
    chunk_count: int


class IngestResponse(BaseModel):
    status: str
    indexed_count: int
    doc_count: int
    chunk_count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _UploadedFileShim:
    """
    Adapts a FastAPI UploadFile to the interface expected by
    IngestionPipeline.run_uploaded_files() — which was originally designed
    for Streamlit's UploadedFile (.name + .getvalue()).
    """

    def __init__(self, filename: str, content: bytes) -> None:
        self.name = filename
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """Return knowledge base statistics and Elasticsearch connectivity."""
    try:
        from ingestion.indexers import ElasticsearchVectorStore
        es_store = ElasticsearchVectorStore()
        es_client = es_store.es
        stats = es_store.get_stats()
        chunk_count = stats.get("doc_count", 0) if stats.get("exists") else 0

        unique_file_count = 0
        if stats.get("exists") and es_client:
            try:
                body: Dict[str, Any] = {
                    "size": 0,
                    "aggs": {
                        "unique_files": {
                            "cardinality": {"field": "file_name"}
                        }
                    },
                }
                res = es_client.search(index=es_store.index_name, body=body)
                unique_file_count = res["aggregations"]["unique_files"]["value"]
            except Exception:
                unique_file_count = 0

        online = bool(es_client and es_client.ping())
        return StatusResponse(
            online=online,
            status_text="Online" if online else "Offline",
            unique_file_count=unique_file_count,
            chunk_count=chunk_count,
        )
    except Exception as exc:
        logger.warning("Status check failed: %s", exc)
        return StatusResponse(
            online=False,
            status_text="Offline",
            unique_file_count=0,
            chunk_count=0,
        )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Execute a RAG query and return the answer with source attribution."""
    pipeline = _get_rag_pipeline()
    model = req.llm_model.strip() or None

    try:
        response = pipeline.query(
            user_query=req.query,
            retrieval_method=req.retrieval_method,
            use_reranking=req.use_reranking,
            use_query_rewriting=req.use_query_rewriting,
            prompt_style=req.prompt_style,
            llm_provider_name=req.llm_provider,
            llm_model=model,
        )
    except Exception as exc:
        logger.error("RAG pipeline error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    sources = [
        SourceItem(
            file_name=doc.file_name,
            page_number=doc.page_number,
            text=doc.text[:300],
        )
        for doc in response.retrieved_docs
    ]

    return ChatResponse(
        answer=response.answer,
        sources=sources,
        conversation_id=response.conversation_id,
        response_time_ms=response.response_time_ms,
        total_tokens=response.total_tokens,
        model=response.model,
        retrieval_method=response.retrieval_method,
        prompt_style=response.prompt_style,
    )


@app.post("/api/feedback", response_model=FeedbackResponse)
async def save_feedback(req: FeedbackRequest) -> FeedbackResponse:
    """Persist thumbs-up/down feedback for a conversation."""
    db = _get_db_repo()
    try:
        feedback_id = db.save_feedback(
            FeedbackRecord(
                conversation_id=req.conversation_id,
                sentiment=req.sentiment,
                comment=req.comment,
            )
        )
    except Exception as exc:
        logger.error("Feedback save error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return FeedbackResponse(
        feedback_id=feedback_id,
        message=f"Feedback recorded: {'👍' if req.sentiment == 'positive' else '👎'}",
    )


@app.post("/api/ingest/files", response_model=IngestResponse)
async def ingest_files(
    files: List[UploadFile] = File(...),
    recreate_index: bool = Form(False),
) -> IngestResponse:
    """Upload files and run the ingestion pipeline."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    shims: List[_UploadedFileShim] = []
    for uf in files:
        content = await uf.read()
        shims.append(_UploadedFileShim(filename=uf.filename or "upload", content=content))

    try:
        pipeline = IngestionPipeline()
        result = pipeline.run_uploaded_files(
            uploaded_files=shims,
            recreate_index=recreate_index,
        )
    except Exception as exc:
        logger.error("Ingestion error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return IngestResponse(
        status=result.get("status", "unknown"),
        indexed_count=result.get("indexed_count", 0),
        doc_count=result.get("doc_count", 0),
        chunk_count=result.get("chunk_count", 0),
    )


@app.post("/api/ingest/folder", response_model=IngestResponse)
async def ingest_folder(
    folder_path: str = Form(...),
    recreate_index: bool = Form(False),
) -> IngestResponse:
    """Run the ingestion pipeline on a server-accessible folder path."""
    clean_path = folder_path.strip(" '\"")
    if not clean_path:
        raise HTTPException(status_code=400, detail="folder_path is required")

    clean_path = os.path.normpath(clean_path)
    if not os.path.isdir(clean_path):
        raise HTTPException(
            status_code=400,
            detail=f"Path does not exist or is not a directory: {clean_path}",
        )

    try:
        pipeline = IngestionPipeline()
        result = pipeline.run(folder_path=clean_path, recreate_index=recreate_index)
    except Exception as exc:
        logger.error("Folder ingestion error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return IngestResponse(
        status=result.get("status", "unknown"),
        indexed_count=result.get("indexed_count", 0),
        doc_count=result.get("doc_count", 0),
        chunk_count=result.get("chunk_count", 0),
    )


@app.get("/api/supported-extensions")
async def supported_extensions() -> Dict[str, List[str]]:
    """Return file extensions supported by the document loader."""
    return {
        "extensions": CompositeDocumentLoader.get_supported_extensions_without_dot()
    }


# ---------------------------------------------------------------------------
# Development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)

