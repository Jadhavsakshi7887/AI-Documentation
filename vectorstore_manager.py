from __future__ import annotations

import json
import math
import os
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import Chroma

from embedding import LocalEmbedding

try:
    from config import CHUNK_OVERLAP, CHUNK_SIZE, VECTORSTORE_PERSIST_DIR
except ImportError:
    # Fallback if config not available
    VECTORSTORE_PERSIST_DIR = Path(__file__).parent / "chroma_vectorstore"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

vectorstore = None
persist_folder = str(VECTORSTORE_PERSIST_DIR)
REGISTRY_PATH = VECTORSTORE_PERSIST_DIR / "doc_registry.json"
STOPWORDS = {
    "the",
    "and",
    "of",
    "to",
    "a",
    "in",
    "for",
    "is",
    "on",
    "that",
    "with",
    "as",
    "by",
    "at",
    "from",
    "or",
    "an",
    "be",
    "this",
    "are",
    "it",
    "was",
    "were",
}


def _load_registry() -> Dict[str, dict]:
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_registry(registry: Dict[str, dict]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))


def _build_filter(doc_ids: Optional[List[str]]):
    if not doc_ids:
        return None
    if len(doc_ids) == 1:
        return {"doc_id": doc_ids[0]}
    return {"doc_id": {"$in": doc_ids}}


def ensure_vectorstore():
    global vectorstore
    if vectorstore is None:
        embedding_wrapper = LocalEmbedding()
        vectorstore = Chroma(
            persist_directory=persist_folder,
            embedding_function=embedding_wrapper,
        )
    return vectorstore


def _chunk_documents(docs, doc_id: str, doc_name: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". "],
    )

    chunks: List[Document] = []
    for i, doc in enumerate(docs):
        page_chunks = splitter.split_text(doc.page_content)
        for j, chunk_text in enumerate(page_chunks):
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "page": i + 1,
                    "chunk": j + 1,
                },
            )
            chunks.append(chunk_doc)
    return chunks


def add_document(file_path: str, original_name: Optional[str] = None) -> Tuple[str, dict]:
    """
    Ingest a document into the vectorstore and registry.
    Returns (doc_id, doc_metadata).
    """
    vs = ensure_vectorstore()
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load()

    if not docs or all(not doc.page_content.strip() for doc in docs):
        raise ValueError("Document appears empty or unreadable.")

    doc_id = str(uuid.uuid4())
    doc_name = original_name or Path(file_path).name

    chunks = _chunk_documents(docs, doc_id=doc_id, doc_name=doc_name)
    vs.add_documents(chunks)
    vs.persist()

    registry = _load_registry()
    registry[doc_id] = {
        "doc_id": doc_id,
        "doc_name": doc_name,
        "file_path": str(file_path),
        "pages": len(docs),
        "chunks": len(chunks),
        "size_bytes": os.path.getsize(file_path) if os.path.exists(file_path) else None,
        "extension": Path(file_path).suffix.lower(),
    }
    _save_registry(registry)
    return doc_id, registry[doc_id]


def list_documents() -> List[dict]:
    registry = _load_registry()
    return list(registry.values())


def search(query: str, k: int = 8, doc_ids: Optional[List[str]] = None):
    vs = ensure_vectorstore()
    where = _build_filter(doc_ids)
    return vs.similarity_search(query, k=k, filter=where)


def keyword_frequency(doc_ids: Optional[List[str]] = None, top_n: int = 15):
    vs = ensure_vectorstore()
    where = _build_filter(doc_ids)
    raw = vs._collection.get(
        where=where,
        include=["documents"],
    )
    documents = raw.get("documents") or []
    text = " ".join(documents)
    tokens = re.findall(r"[A-Za-z]{3,}", text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS]
    counts = Counter(tokens)
    return counts.most_common(top_n)


def get_chunks(
    doc_ids: Optional[List[str]] = None, limit: int = 200
) -> List[dict]:
    """
    Return chunk snippets with metadata for UI display.
    """
    vs = ensure_vectorstore()
    where = _build_filter(doc_ids)
    raw = vs._collection.get(
        where=where,
        include=["metadatas", "documents"],
        limit=limit,
    )
    metadatas = raw.get("metadatas") or []
    documents = raw.get("documents") or []
    results = []
    for meta, doc in zip(metadatas, documents):
        results.append(
            {
                "doc_id": meta.get("doc_id"),
                "doc_name": meta.get("doc_name"),
                "page": meta.get("page"),
                "chunk": meta.get("chunk"),
                "text": doc,
            }
        )
    return results


def similarity_search_with_sources(
    query: str, k: int = 8, doc_ids: Optional[List[str]] = None
):
    """Helper to get chunks plus metadata for RAG."""
    return search(query=query, k=k, doc_ids=doc_ids)


def _doc_text_snapshot(doc_id: str, max_chars: int = 4000) -> Optional[str]:
    vs = ensure_vectorstore()
    raw = vs._collection.get(
        where={"doc_id": doc_id},
        include=["documents"],
    )
    documents = raw.get("documents") or []
    if not documents:
        return None
    return " ".join(documents)[:max_chars]


def doc_similarity_matrix(doc_ids: List[str]) -> dict:
    """
    Compute a cosine similarity matrix between documents using averaged embeddings.
    """
    from embedding import LocalEmbedding  # local import to avoid cycles

    registry = _load_registry()
    texts = []
    labels = []

    for doc_id in doc_ids:
        text = _doc_text_snapshot(doc_id)
        if text:
            texts.append(text)
            labels.append(registry.get(doc_id, {}).get("doc_name", doc_id))

    if not texts:
        return {"labels": [], "matrix": []}

    embedder = LocalEmbedding()
    embeddings = embedder.embed_documents(texts)

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    size = len(embeddings)
    matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            matrix[i][j] = cosine(embeddings[i], embeddings[j])

    return {"labels": labels, "matrix": matrix}


def clear_registry():
    """Utility for tests: clear registry and vectorstore."""
    global vectorstore
    if REGISTRY_PATH.exists():
        REGISTRY_PATH.unlink()
    if VECTORSTORE_PERSIST_DIR.exists():
        for item in VECTORSTORE_PERSIST_DIR.iterdir():
            if item.is_file():
                item.unlink()
    vectorstore = None
