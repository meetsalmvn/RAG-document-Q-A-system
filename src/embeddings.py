"""
Wrapper around Google's Gemini embedding API.
"""
import time
from typing import List
import google.generativeai as genai

EMBEDDING_MODEL = "models/gemini-embedding-001"


def embed_texts(
    texts: List[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = 10,
    max_retries: int = 3,
) -> List[List[float]]:
    """
    Embed a list of texts using Gemini's embedding model.

    task_type: "retrieval_document" for indexing, "retrieval_query" for search queries.
    Batches requests and retries on transient failures (e.g. rate limits).
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        for attempt in range(max_retries):
            try:
                result = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=batch,
                    task_type=task_type,
                )
                all_embeddings.extend(result["embedding"])
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to embed batch after {max_retries} attempts: {e}"
                    )
                time.sleep(2 ** attempt)  # exponential backoff

    return all_embeddings


def embed_query(query: str) -> List[float]:
    """Embed a single search query."""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="RETRIEVAL_QUERY",
    )
    return result["embedding"]
