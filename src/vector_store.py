"""
Local vector store built on ChromaDB (no external DB account needed).
"""
from typing import List, Dict
import streamlit as st
import chromadb
from chromadb.config import Settings
from src.embeddings import embed_texts, embed_query


@st.cache_resource(show_spinner=False)
def _get_chroma_client(persist_dir: str):
    """
    Create (and cache) a single ChromaDB client for the lifetime of the app process.

    Creating more than one PersistentClient pointed at the same directory in the
    same process can corrupt ChromaDB's internal Rust binding cleanup, causing
    'RustBindingsAPI object has no attribute bindings' after the app sleeps and
    wakes up (Streamlit Cloud) or across reruns. Caching with st.cache_resource
    guarantees exactly one client is ever created per process.
    """
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )


class VectorStore:
    def __init__(self, persist_dir: str = "chroma_db", collection_name: str = "pdf_chunks"):
        self.client = _get_chroma_client(persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def is_empty(self) -> bool:
        return self.collection.count() == 0

    def clear(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(name=self.collection.name)

    def add_chunks(self, chunks: List[Dict], batch_size: int = 10):
        """
        Embed and store chunks. Each chunk dict needs: chunk_id, text, page, source.
        """
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts, task_type="retrieval_document", batch_size=batch_size)

        self.collection.add(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"page": c["page"], "source": c["source"]} for c in chunks],
        )

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Return the top_k most relevant chunks for a query.
        """
        query_embedding = embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            hits.append({
                "text": doc,
                "page": meta["page"],
                "source": meta["source"],
                "score": 1 - dist,  # convert distance to a similarity-like score
            })

        return hits