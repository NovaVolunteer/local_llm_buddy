# local_llm_buddy

A Python package that loads [Otter.ai](https://otter.ai) meeting transcripts,
vectorises them with a local [Ollama](https://ollama.com) embedding model,
stores them in [Pinecone](https://www.pinecone.io), and answers natural-language
questions about your meetings using a **local open-weight LLM** (via Ollama) in
a **Retrieval-Augmented Generation (RAG)** architecture powered by
[LangChain](https://python.langchain.com).

A [Streamlit](https://streamlit.io) web interface is included so you can chat
with your transcripts right from the browser.

---

## Architecture

```
Otter.ai .txt / .srt
        │
        ▼
OtterTranscriptLoader   (loader.py)
        │  splits into speaker-turn Documents
        ▼
RecursiveCharacterTextSplitter
        │  chunks
        ▼
OllamaEmbeddings        (nomic-embed-text or similar)
        │  vectors
        ▼
Pinecone Index          (vectorstore.py)
        │
        │  similarity search at query time
        ▼
LangChain RAG Chain     (rag.py)
        │  retrieved chunks → ChatPromptTemplate
        ▼
ChatOllama              (llama3, mistral, …)
        │
        ▼
Streamlit UI            (app.py)
```

---

## Prerequisites

| Tool | Purpose |
|------|---------|
| [Ollama](https://ollama.com) | Runs the LLM and embedding model locally |
| [Pinecone account](https://app.pinecone.io) | Cloud vector store |
| Python ≥ 3.10 | Runtime |

### Pull the Ollama models you want to use

```bash
ollama pull llama3            # LLM
ollama pull nomic-embed-text  # embedding model (768-dim)
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/NovaVolunteer/local_llm_buddy.git
cd local_llm_buddy

# Install the package (editable mode recommended for development)
pip install -e .
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `PINECONE_API_KEY` | *(required)* | Your Pinecone API key |
| `PINECONE_INDEX_NAME` | `otter-transcripts` | Pinecone index name |
| `PINECONE_ENVIRONMENT` | `us-east-1-aws` | Pinecone cloud region |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Running Ollama instance |
| `OLLAMA_MODEL` | `llama3` | LLM model tag |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model tag |
| `EMBED_DIMENSION` | `768` | Embedding vector size |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVER_K` | `4` | Chunks retrieved per query |

> **`EMBED_DIMENSION` must match your embedding model.**  
> `nomic-embed-text` → 768 · `mxbai-embed-large` → 1024

---

## Streamlit UI

```bash
streamlit run app.py
```

1. Enter your Pinecone API key in the sidebar and click **Apply Settings**.
2. Upload one or more Otter.ai exports (`.txt` or `.srt`) and click **Ingest Transcripts**.
3. Ask questions in the chat box.

---

## Python API

```python
from local_llm_buddy import (
    Settings,
    OtterTranscriptLoader,
    PineconeStore,
    build_rag_chain,
)

# 1. Load transcripts
loader = OtterTranscriptLoader(["meeting_a.txt", "meeting_b.srt"])
docs = loader.load()

# 2. Ingest into Pinecone (creates the index if it doesn't exist)
settings = Settings()
store = PineconeStore(settings)
n_chunks = store.ingest(docs)
print(f"Upserted {n_chunks} chunks")

# 3. Build the RAG chain and query
chain = build_rag_chain(store.retriever(), settings)
answer = chain.invoke("What did Alice say about the Q3 roadmap?")
print(answer)
```

### Transcript file formats

**Plain-text Otter.ai export** (`.txt`)  
Each speaker turn begins with a line of the form `Speaker Name  HH:MM:SS`:

```
Alice  00:00:05
Hello everyone, welcome to the Q3 planning meeting.

Bob  00:00:12
Thanks Alice. Let me share my screen.
```

**SRT subtitles** (`.srt`)  
Standard SRT format exported from Otter.ai or any subtitle editor.

---

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

---

## Project layout

```
local_llm_buddy/
├── local_llm_buddy/
│   ├── __init__.py      # public API
│   ├── __main__.py      # python -m local_llm_buddy
│   ├── config.py        # Settings (env vars / .env)
│   ├── loader.py        # OtterTranscriptLoader
│   ├── vectorstore.py   # PineconeStore
│   └── rag.py           # build_rag_chain
├── tests/
│   └── test_loader_and_config.py
├── app.py               # Streamlit UI
├── pyproject.toml
├── requirements.txt
└── .env.example
```
