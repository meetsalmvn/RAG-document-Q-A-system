"""
Streamlit UI for the PDF RAG Q&A system.
Run with: streamlit run app.py
"""
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

from src.pdf_loader import extract_text_from_pdf
from src.chunking import chunk_pages
from src.vector_store import VectorStore
from src.rag_pipeline import answer_question

load_dotenv()

st.set_page_config(page_title="PDF Q&A with Gemini", page_icon="📄", layout="wide")


def get_api_key() -> str:
    """Get API key from env var, Streamlit secrets, or sidebar input."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        try:
            key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            key = None
    return key


def init_session_state():
    if "store" not in st.session_state:
        st.session_state.store = VectorStore()
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def main():
    init_session_state()

    st.title("📄 PDF Q&A — RAG with Gemini")
    st.caption("Upload PDFs, then ask questions answered strictly from their content, with page citations.")

    with st.sidebar:
        st.header("Setup")

        api_key = get_api_key()
        if api_key:
            st.success("Gemini API key loaded from server secrets.")
            api_key_input = api_key
            genai.configure(api_key=api_key_input)
        else:
            api_key_input = st.text_input(
                "Gemini API Key",
                type="password",
                help="Get a free key at https://aistudio.google.com/apikey",
            )
            if api_key_input:
                genai.configure(api_key=api_key_input)

        st.divider()
        st.header("Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files", type=["pdf"], accept_multiple_files=True
        )

        chunk_size = st.slider("Chunk size (characters)", 500, 2000, 1000, 100)
        top_k = st.slider("Chunks to retrieve per question", 3, 10, 5)

        if st.button("Process PDFs", type="primary", disabled=not uploaded_files):
            if not api_key_input:
                st.error("Please enter your Gemini API key first.")
            else:
                with st.spinner("Extracting and indexing text..."):
                    all_chunks = []
                    for uf in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uf.read())
                            tmp_path = tmp.name

                        pages = extract_text_from_pdf(tmp_path)
                        chunks = chunk_pages(pages, chunk_size=chunk_size)
                        all_chunks.extend(chunks)
                        os.unlink(tmp_path)

                    if all_chunks:
                        st.session_state.store.add_chunks(all_chunks)
                        st.session_state.indexed_files.extend([f.name for f in uploaded_files])
                        st.success(f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} file(s).")
                    else:
                        st.warning("No extractable text found (PDFs may be scanned images without OCR).")

        if st.session_state.indexed_files:
            st.divider()
            st.subheader("Indexed files")
            for f in st.session_state.indexed_files:
                st.text(f"✓ {f}")

            if st.button("Clear index"):
                st.session_state.store.clear()
                st.session_state.indexed_files = []
                st.session_state.chat_history = []
                st.rerun()

    # Chat interface
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"**{s['source']}, page {s['page']}** (score: {s['score']:.2f})")
                        st.caption(s["text"][:300] + "...")

    question = st.chat_input("Ask a question about your PDFs...")

    if question:
        if not api_key_input:
            st.error("Please enter your Gemini API key in the sidebar.")
        elif not st.session_state.indexed_files:
            st.error("Please upload and process at least one PDF first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer, sources = answer_question(question, st.session_state.store, top_k=top_k)
                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                            for s in sources:
                                st.markdown(f"**{s['source']}, page {s['page']}** (score: {s['score']:.2f})")
                                st.caption(s["text"][:300] + "...")

            st.session_state.chat_history.append({
                "role": "assistant", "content": answer, "sources": sources
            })


if __name__ == "__main__":
    main()
