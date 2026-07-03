"""
Retrieval-Augmented Generation pipeline: retrieve relevant chunks,
then ask Gemini to answer grounded strictly in that context.
"""
from typing import List, Dict, Tuple
import google.generativeai as genai
from src.vector_store import VectorStore

GENERATION_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a set of PDF documents.
Rules:
- Answer ONLY using the provided context. Do not use outside knowledge.
- If the context doesn't contain the answer, say so clearly instead of guessing.
- Cite the source and page number for each claim you make, like (source.pdf, p.3).
- Be concise and direct.
"""


def build_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into a context block for the prompt."""
    blocks = []
    for c in chunks:
        blocks.append(f"[{c['source']}, page {c['page']}]\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


def answer_question(
    question: str,
    store: VectorStore,
    top_k: int = 5,
) -> Tuple[str, List[Dict]]:
    """
    Run the full RAG flow: retrieve -> build prompt -> generate answer.

    Returns (answer_text, retrieved_chunks) so the UI can show sources.
    """
    retrieved = store.search(question, top_k=top_k)

    if not retrieved:
        return "No documents have been indexed yet. Please upload and process a PDF first.", []

    context = build_context(retrieved)

    prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question: {question}

Answer:"""

    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content(prompt)

    return response.text, retrieved
