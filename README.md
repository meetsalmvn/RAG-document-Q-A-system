# 📄 PDF Q&A — RAG System with Gemini

A Retrieval-Augmented Generation (RAG) system that answers questions about your PDF documents — research papers, manuals, textbooks, reports — using Google's free-tier Gemini API. Every answer is grounded strictly in the uploaded documents and cited with source + page number, reducing hallucination compared to asking an LLM directly.

**[Add your deployed Streamlit Cloud link here]**

![Demo1](docs/demo1.gif) <!-- record a short screen capture and add it here -->
![Demo2](docs/demo2.gif)

## Why this project

Off-the-shelf LLMs don't know about your private or niche documents, and they hallucinate confidently when they don't know an answer. This project solves that with a standard RAG architecture:

1. **Ingest** — extract text from uploaded PDFs, page by page
2. **Chunk** — split text into overlapping windows so context isn't lost at boundaries
3. **Embed** — convert each chunk into a vector using Gemini's embedding model
4. **Store** — index vectors locally in ChromaDB (no external database account needed)
5. **Retrieve** — on each question, find the most semantically relevant chunks
6. **Generate** — ask Gemini to answer *using only the retrieved context*, with citations

## Architecture

```
┌─────────┐    ┌───────────┐    ┌────────────┐    ┌───────────────┐
│  PDFs   │───▶│  Chunking │───▶│  Gemini     │───▶│  ChromaDB      │
│ (upload)│    │ (overlap) │    │ Embeddings  │    │ (vector store) │
└─────────┘    └───────────┘    └────────────┘    └───────┬───────┘
                                                            │
┌─────────┐    ┌───────────┐    ┌────────────┐            │
│  Answer │◀───│  Gemini   │◀───│  Top-k      │◀───────────┘
│ + cites │    │ Generate  │    │  Retrieval  │  (query embedding)
└─────────┘    └───────────┘    └────────────┘
```

## Tech stack

| Component | Choice | Why |
|---|---|---|
| LLM + Embeddings | Google Gemini API | Genuinely free tier — no credit card needed |
| Vector store | ChromaDB (local, persistent) | Zero-config, no external DB account |
| PDF parsing | pypdf | Lightweight, reliable text extraction |
| UI | Streamlit | Fast to build, easy to deploy free |

## Project structure

```
pdf-rag-gemini/
├── app.py                 # Streamlit UI — entry point
├── requirements.txt
├── .env.example
├── src/
│   ├── pdf_loader.py       # PDF text extraction
│   ├── chunking.py         # Overlapping text chunking
│   ├── embeddings.py       # Gemini embedding API wrapper (batched, retried)
│   ├── vector_store.py     # ChromaDB wrapper (index + search)
│   └── rag_pipeline.py     # Retrieval + grounded generation
└── sample_data/            # Drop sample PDFs here for quick testing
```

## Setup

### 1. Get a free Gemini API key
Go to [Google AI Studio](https://aistudio.google.com/apikey) and create a free API key (no billing required for the free tier).

### 2. Install dependencies
```bash
git clone https://github.com/<your-username>/pdf-rag-gemini.git
cd pdf-rag-gemini
pip install -r requirements.txt
```

### 3. Configure your API key
Copy `.env.example` to `.env` and add your key:
```bash
cp .env.example .env
# edit .env and set GEMINI_API_KEY=your_actual_key
```
(You can also paste the key directly into the app's sidebar at runtime instead.)

### 4. Run
```bash
streamlit run app.py
```
Open the local URL Streamlit prints (usually `http://localhost:8501`), upload a PDF, click **Process PDFs**, and start asking questions.

## Design decisions worth mentioning in an interview

- **Chunking with overlap** (150 chars by default): prevents an answer's supporting sentence from being split exactly at a chunk boundary and becoming unretrievable.
- **Retry with exponential backoff on embedding calls**: the free tier has rate limits; batched requests with backoff avoid failing on large PDFs.
- **Grounded generation via prompt constraints**: the system prompt explicitly instructs the model to answer only from retrieved context and say "I don't know" rather than guess — this is what separates a RAG system from just calling an LLM.
- **Local vector store**: avoids needing a hosted vector DB account (Pinecone, etc.), which keeps the project free and easy for anyone to clone and run.

## Known limitations (good to mention proactively)

- Scanned PDFs (images of text, not real text) won't extract — would need OCR (e.g. Tesseract) as a fallback, which is a natural "future work" extension.
- ChromaDB may print a harmless telemetry warning on startup in some versions — it doesn't affect functionality.
- Retrieval quality depends on chunk size; very short or very long chunks both hurt relevance — the sidebar exposes this as a tunable parameter so you can demonstrate the tradeoff live.

## Possible extensions

- Add OCR fallback for scanned documents
- Support multi-turn conversational memory (follow-up questions referencing prior answers)
- Swap ChromaDB for a hosted vector DB and deploy multi-user
- Add evaluation harness (e.g. compare retrieval@k against a labeled Q&A set)

## Deployment

This app deploys for free on [Streamlit Community Cloud](https://streamlit.io/cloud):
1. Push this repo to GitHub
2. Go to share.streamlit.io, connect your repo, set `app.py` as the entry point
3. Add `GEMINI_API_KEY` as a secret in the app settings

## License

MIT
