"""
Streamlit UI for the PDF RAG Q&A system.
Run with: streamlit run app.py
"""
import os
import tempfile
import time
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

from src.pdf_loader import extract_text_from_pdf
from src.chunking import chunk_pages
from src.vector_store import VectorStore
from src.rag_pipeline import answer_question

load_dotenv()

st.set_page_config(page_title="PDF AI Assistant — RAG with Gemini", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    .stat-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 8px;
    }
    .stat-card .stat-label {
        font-size: 0.8rem;
        opacity: 0.65;
        margin-bottom: 4px;
    }
    .stat-card .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .stat-card .stat-sub {
        font-size: 0.75rem;
        opacity: 0.5;
        margin-top: 2px;
    }
    .source-pill {
        display: inline-block;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 8px 12px;
        margin: 4px 6px 4px 0;
        font-size: 0.82rem;
        vertical-align: top;
    }
    .source-pill .source-name {
        font-weight: 600;
    }
    .source-pill .source-meta {
        opacity: 0.6;
        font-size: 0.75rem;
    }
    .index-status-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 14px 16px;
    }
    .app-header-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .app-header-sub {
        opacity: 0.6;
        font-size: 0.9rem;
        margin-top: 2px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def get_api_key() -> str:
    """Get API key from env var or Streamlit secrets."""
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
    if "questions_asked" not in st.session_state:
        st.session_state.questions_asked = 0
    if "last_updated" not in st.session_state:
        st.session_state.last_updated = None
    if "total_chunks" not in st.session_state:
        st.session_state.total_chunks = 0


def render_stat_cards():
    col1, col2, col3, col4 = st.columns(4)
    last_updated_str = "—"
    if st.session_state.last_updated:
        mins_ago = int((time.time() - st.session_state.last_updated) / 60)
        last_updated_str = "just now" if mins_ago < 1 else f"{mins_ago}m ago"

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">📄 Documents</div>
            <div class="stat-value">{len(st.session_state.indexed_files)}</div>
            <div class="stat-sub">PDF files uploaded</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">🧩 Chunks</div>
            <div class="stat-value">{st.session_state.total_chunks}</div>
            <div class="stat-sub">Text chunks indexed</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">💬 Questions Asked</div>
            <div class="stat-value">{st.session_state.questions_asked}</div>
            <div class="stat-sub">Total this session</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">🕒 Last Updated</div>
            <div class="stat-value" style="font-size:1.1rem;">{last_updated_str}</div>
            <div class="stat-sub">Index last updated</div>
        </div>
        """, unsafe_allow_html=True)


def render_sources(sources):
    st.markdown("**📖 Sources**")
    pills_html = ""
    for s in sources:
        pills_html += f"""
        <span class="source-pill">
            <span class="source-name">📄 {s['source']}</span><br>
            <span class="source-meta">Page {s['page']} · score {s['score']:.2f}</span>
        </span>
        """
    st.markdown(pills_html, unsafe_allow_html=True)
    with st.expander("View source text excerpts"):
        for s in sources:
            st.markdown(f"**{s['source']}, page {s['page']}**")
            st.caption(s["text"][:300] + "...")


def main():
    init_session_state()

    # Header
    st.markdown("""
    <div class="app-header-title">✨ Ask anything about your PDFs</div>
    <div class="app-header-sub">Get answers from your documents with AI — grounded, cited, and hallucination-checked.</div>
    <br>
    """, unsafe_allow_html=True)

    render_stat_cards()
    st.markdown("<br>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 📄 PDF AI Assistant")
        st.caption("RAG with Gemini")
        st.divider()

        st.markdown("#### Setup")
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
        st.markdown("#### Upload Documents")
        uploaded_files = st.file_uploader(
            "Drag & drop PDF files here", type=["pdf"], accept_multiple_files=True,
            help="Max file size: 200MB per file"
        )

        chunk_size = st.slider("Chunk size (characters)", 500, 2000, 1000, 100)
        top_k = st.slider("Chunks to retrieve per question", 3, 10, 5)

        if st.button("Process PDFs", type="primary", disabled=not uploaded_files, use_container_width=True):
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
                        st.session_state.total_chunks += len(all_chunks)
                        st.session_state.last_updated = time.time()
                        st.success(f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} file(s).")
                    else:
                        st.warning("No extractable text found (PDFs may be scanned images without OCR).")

        st.divider()
        st.markdown("#### Index Status")
        st.markdown(f"""
        <div class="index-status-card">
            📄 Files Indexed: <b>{len(st.session_state.indexed_files)}</b><br>
            🧩 Chunks Indexed: <b>{st.session_state.total_chunks}</b>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.indexed_files:
            with st.expander("Indexed files"):
                for f in st.session_state.indexed_files:
                    st.text(f"✓ {f}")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Index", use_container_width=True):
            st.session_state.store.clear()
            st.session_state.indexed_files = []
            st.session_state.chat_history = []
            st.session_state.total_chunks = 0
            st.session_state.questions_asked = 0
            st.session_state.last_updated = None
            st.rerun()

    # Chat interface
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])

    question = st.chat_input("Ask a question about your PDFs...")
    st.caption("Answers are generated strictly from your uploaded documents.")

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
                        render_sources(sources)

            st.session_state.chat_history.append({
                "role": "assistant", "content": answer, "sources": sources
            })
            st.session_state.questions_asked += 1


if __name__ == "__main__":
    main()