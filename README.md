#Install
https://eshap09.github.io/PDF-Rag/
brew install Eshap09/rag

# RAG API

A production-grade Retrieval Augmented Generation (RAG) API built with FastAPI, LangChain, ChromaDB, and Groq. Upload documents, ask questions, get answers grounded in your data — not hallucinations.

## What it does

- Upload PDF, TXT, or Markdown files via REST API or UI
- Automatically chunks, embeds, and indexes documents into ChromaDB
- Hybrid retrieval — semantic search + cross-encoder reranking
- OCR fallback for image-based PDFs (via pytesseract)
- Answers grounded strictly in your documents, with source citations and page numbers
- Re-ingesting the same file replaces old chunks — no duplicates

## Tech stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| LLM | Groq (llama-3.3-70b-versatile) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2, local) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector store | ChromaDB (Docker or embedded) |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| OCR | pytesseract + pdf2image |
| Config | Pydantic Settings |
| UI | Vanilla HTML/CSS/JS served by FastAPI |

## Project structure

```
rag-api/
├── app/
│   ├── main.py              # FastAPI app, lifespan, middleware
│   ├── config.py            # All config via env vars (Pydantic Settings)
│   ├── exceptions.py        # Custom HTTP exceptions
│   ├── ui/
│   │   └── index.html       # Web UI (served at /)
│   ├── api/
│   │   ├── schemas.py       # Request/response Pydantic models
│   │   └── v1/
│   │       ├── router.py    # v1 router (/api/v1/...)
│   │       └── routes/
│   │           ├── ingest.py
│   │           └── query.py
│   └── rag/
│       ├── chunker.py       # Document loading + chunking (text + OCR)
│       ├── embedder.py      # HuggingFace embeddings (local, no API cost)
│       ├── vectorstore.py   # ChromaDB wrapper
│       ├── retriever.py     # Hybrid search + reranking
│       └── pipeline.py      # Orchestrates ingest and query flows
├── notebooks/
│   └── rag_explorer.ipynb   # Jupyter notebook for exploration
├── data/                    # Place your documents here
├── requirements.txt
└── .env
```

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/YOUR_USERNAME/rag-api.git
cd rag-api
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install system dependencies (for OCR)

```bash
# macOS
brew install tesseract poppler

# Ubuntu/Debian
sudo apt install tesseract-ocr poppler-utils
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Start ChromaDB (choose one)

**Option A — Docker (recommended):**

```bash
docker run -d \
  --name chromadb \
  -p 8001:8000 \
  -v chroma-data:/chroma/chroma \
  chromadb/chroma:latest
```

**Option B — Embedded (no Docker needed):**

Set in `.env`:
```
CHROMA_HOST=embedded
```

### 6. Start the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` for the UI or `http://localhost:8000/docs` for the API explorer.

## API endpoints

### `POST /api/v1/ingest`

Upload a document to index.

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@your_document.pdf"
```

Response:
```json
{
  "source": "your_document.pdf",
  "chunks_added": 42,
  "replaced": false,
  "message": "Document indexed successfully"
}
```

### `POST /api/v1/query`

Ask a question about indexed documents.

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main topics covered?"}'
```

Response:
```json
{
  "answer": "The document covers...",
  "sources": [
    {
      "source": "your_document.pdf",
      "chunk": 3,
      "page_number": 2,
      "extraction": "text",
      "content": "..."
    }
  ]
}
```

### `DELETE /api/v1/collection`

Wipe all indexed documents.

```bash
curl -X DELETE http://localhost:8000/api/v1/collection
```

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

## Configuration

All config is via environment variables. Defaults work out of the box.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | required | Your Groq API key |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model |
| `LLM_TEMPERATURE` | `0.1` | Lower = more factual |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `CHUNK_SIZE` | `500` | Max chars per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `5` | Chunks returned per query |
| `RETRIEVAL_CANDIDATE_K` | `20` | Candidates before reranking |
| `USE_RERANKER` | `true` | Enable cross-encoder reranking |
| `CHROMA_HOST` | `localhost` | ChromaDB host |
| `CHROMA_PORT` | `8001` | ChromaDB port |
| `CHROMA_COLLECTION` | `rag_documents` | Collection name |
| `MAX_UPLOAD_MB` | `50` | Max file size |

## How it works

### Indexing pipeline

```
File upload → load_and_chunk() → embed_documents() → vectorstore.add()
                    ↓
          text extraction (pypdf)
          if sparse → OCR (pytesseract)
          → RecursiveCharacterTextSplitter
          → Chunk objects with page_number metadata
```

### Query pipeline

```
Question → embed_query() → similarity_search(top 20) → rerank(top 5) → LLM → answer
```

The reranker (cross-encoder) reads each `(question, chunk)` pair together and rescores by actual relevance — much more accurate than vector similarity alone.

### Why the answer is grounded

The system prompt instructs the LLM to answer using **only** the retrieved context. If the answer isn't in the documents, it says so instead of hallucinating.

## Supported file types

| Format | Extraction | OCR fallback |
|---|---|---|
| `.pdf` | pypdf text layer | yes (pytesseract) |
| `.txt` | direct read | no |
| `.md` | direct read | no |

## Running the Jupyter notebook

```bash
pip install jupyter
jupyter notebook
```

Open `notebooks/rag_explorer.ipynb` to inspect chunks, embeddings, ChromaDB contents, and run queries interactively.

## Get a Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up and create an API key
3. Add it to your `.env` file

Groq offers a generous free tier — llama-3.3-70b-versatile is fast and free for development.
