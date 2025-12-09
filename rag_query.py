from __future__ import annotations

from typing import List, Optional

import requests
import vectorstore_manager

try:
    from config import RAG_API_KEY, RAG_API_URL, RAG_MODEL
except ImportError:
    # Fallback if config not available
    RAG_API_KEY = "sk-voidai-zAFjyYrdhOfQwZahZFXpJm4U3DC4wni8BgbWyxi99B05kiIxkCitl3LjE6z09DBsCNuJnUE0JAGwcZleA4BGJOVaAnHWp6TuHQRQIRQK5EEnuHCRBTGh0hpSDT3kib_LkY19HQ"
    RAG_API_URL = "https://api.voidai.app/v1/chat/completions"
    RAG_MODEL = "gpt-4o"

API_KEY = RAG_API_KEY
API_URL = RAG_API_URL
MODEL = RAG_MODEL


def rag_query(
    question: str, doc_ids: Optional[List[str]] = None, k: int = 8
) -> dict:
    """
    Performs a RAG query with citations using the vectorstore for context retrieval.
    Returns a dict: {answer, sources, success}.
    """
    # 1️⃣ Retrieve relevant chunks
    retrieved_docs = vectorstore_manager.similarity_search_with_sources(
        question, k=k, doc_ids=doc_ids
    )
    if not retrieved_docs:
        return {
            "answer": "No relevant documents found.",
            "sources": [],
            "success": False,
        }

    context_lines = []
    sources = []
    for idx, doc in enumerate(retrieved_docs, start=1):
        meta = doc.metadata or {}
        doc_name = meta.get("doc_name", "document")
        page = meta.get("page", "?")
        snippet = doc.page_content.replace("\n", " ").strip()
        context_lines.append(f"[{idx}] {doc_name} | p.{page}: {snippet}")
        sources.append(
            {
                "id": idx,
                "doc_id": meta.get("doc_id"),
                "doc_name": doc_name,
                "page": page,
                "chunk": meta.get("chunk"),
                "snippet": snippet[:500],
            }
        )

    context = "\n".join(context_lines)

    # 2️⃣ Build the API request payload
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are DocuVision AI. Answer ONLY using the provided sources. "
                    "Do not add external information. "
                    "Cite sources inline using the bracket id, e.g., [1], [2]. "
                    "If the answer is not present, reply exactly: 'Not found in the provided documents.'"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Sources:\n{context}\n\nQuestion: {question}\n\n"
                    "Answer concisely and include citations like [1], [2]."
                ),
            },
        ],
        "max_tokens": 400,
        "temperature": 0.2,
        "top_p": 0.9,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    # 3️⃣ Send request to LLM
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        message = data.get("choices", [{}])[0].get("message", {})
        answer = (message.get("content", "") or "").strip()
        return {
            "answer": answer or "No content returned from model.",
            "sources": sources,
            "success": True,
        }

    except requests.exceptions.HTTPError:
        detail = ""
        try:
            detail = response.text
        except Exception:
            detail = ""
        return {
            "answer": f"HTTP error: {detail}",
            "sources": sources,
            "success": False,
        }
    except Exception as e:
        return {
            "answer": f"Error during query: {str(e)}",
            "sources": sources,
            "success": False,
        }
