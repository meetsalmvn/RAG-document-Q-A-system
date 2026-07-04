"""
Chunk extracted page text into overlapping windows suitable for embedding.
"""
import uuid
from typing import List, Dict


def chunk_pages(
    pages: List[Dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Dict]:
    """
    Split page-level text into overlapping chunks.

    Each input item: {"page": int, "text": str, "source": str}
    Each output item adds: {"chunk_id": str}

    Overlap preserves context across chunk boundaries so an answer
    that straddles two chunks isn't lost.

    chunk_id is a UUID rather than a sequential counter so IDs stay
    unique across multiple calls (e.g. processing PDFs in separate
    batches against a persisted vector store).
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []

    for page in pages:
        text = page["text"]
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": f"chunk_{uuid.uuid4().hex}",
                    "text": chunk_text,
                    "page": page["page"],
                    "source": page["source"],
                })

            if end == text_len:
                break
            start = end - chunk_overlap

    return chunks