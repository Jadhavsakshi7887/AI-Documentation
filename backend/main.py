"""
DocuVision AI backend (FastAPI)

Features:
- Multi-document upload and ingestion
- RAG Q&A with citations
- Keyword search and chunk browsing
- Keyword frequency stats
- Document similarity matrix for heatmaps
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path to import project modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import vectorstore_manager
from rag_query import rag_query

app = FastAPI(title="DocuVision AI API", version="2.0.0")

# CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure temp_uploads directory exists
TEMP_UPLOADS_DIR = project_root / "temp_uploads"
TEMP_UPLOADS_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".doc", ".docx"}


# ---------- Pydantic models ----------
class UploadResult(BaseModel):
    doc_id: str
    doc_name: str
    pages: int
    chunks: int
    size_bytes: Optional[int] = None
    extension: Optional[str] = None
    file_path: str


class UploadResponse(BaseModel):
    success: bool
    documents: List[UploadResult]


class QueryRequest(BaseModel):
    question: str
    doc_ids: Optional[List[str]] = None
    top_k: int = 8


class SourceItem(BaseModel):
    id: int
    doc_id: Optional[str]
    doc_name: Optional[str]
    page: Optional[int]
    chunk: Optional[int]
    snippet: Optional[str]


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    success: bool = True


class SearchRequest(BaseModel):
    query: str
    doc_ids: Optional[List[str]] = None
    top_k: int = 10


class SearchMatch(BaseModel):
    doc_id: Optional[str]
    doc_name: Optional[str]
    page: Optional[int]
    chunk: Optional[int]
    snippet: str


class SearchResponse(BaseModel):
    matches: List[SearchMatch]


class StatsRequest(BaseModel):
    doc_ids: Optional[List[str]] = None
    top_n: int = 15


class KeywordStat(BaseModel):
    term: str
    count: int


class KeywordStatsResponse(BaseModel):
    keywords: List[KeywordStat]


class SimilarityRequest(BaseModel):
    doc_ids: Optional[List[str]] = None


class SimilarityResponse(BaseModel):
    labels: List[str]
    matrix: List[List[float]]


# ---------- Routes ----------
@app.get("/")
async def root():
    docs = vectorstore_manager.list_documents()
    return {
        "message": "DocuVision AI API is running",
        "status": "healthy",
        "documents": len(docs),
    }


@app.get("/documents", response_model=List[UploadResult])
async def list_docs():
    return vectorstore_manager.list_documents()


@app.post("/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    processed: List[dict] = []
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type {ext} not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
            )

        save_path = TEMP_UPLOADS_DIR / file.filename
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)

        try:
            doc_id, meta = vectorstore_manager.add_document(
                str(save_path), original_name=file.filename
            )
            processed.append(meta)
        except Exception as e:
            if save_path.exists():
                save_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=500, detail=f"Error processing {file.filename}: {str(e)}"
            )

    return UploadResponse(success=True, documents=processed)


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = rag_query(
        question=request.question, doc_ids=request.doc_ids, k=request.top_k
    )

    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("answer", "Query failed"))

    return QueryResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        success=True,
    )


@app.post("/search", response_model=SearchResponse)
async def search_chunks(request: SearchRequest):
    matches_raw = vectorstore_manager.search(
        query=request.query, k=request.top_k, doc_ids=request.doc_ids
    )
    matches: List[SearchMatch] = []
    for doc in matches_raw:
        meta = doc.metadata or {}
        snippet = doc.page_content.replace("\n", " ").strip()
        matches.append(
            SearchMatch(
                doc_id=meta.get("doc_id"),
                doc_name=meta.get("doc_name"),
                page=meta.get("page"),
                chunk=meta.get("chunk"),
                snippet=snippet[:500],
            )
        )
    return SearchResponse(matches=matches)


@app.post("/stats/keywords", response_model=KeywordStatsResponse)
async def keyword_stats(request: StatsRequest):
    stats = vectorstore_manager.keyword_frequency(
        doc_ids=request.doc_ids, top_n=request.top_n
    )
    keywords = [KeywordStat(term=term, count=count) for term, count in stats]
    return KeywordStatsResponse(keywords=keywords)


@app.post("/stats/similarity", response_model=SimilarityResponse)
async def similarity_matrix(request: SimilarityRequest):
    doc_ids = request.doc_ids
    if not doc_ids:
        doc_ids = [doc["doc_id"] for doc in vectorstore_manager.list_documents()]

    matrix = vectorstore_manager.doc_similarity_matrix(doc_ids)
    return SimilarityResponse(labels=matrix.get("labels", []), matrix=matrix.get("matrix", []))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
