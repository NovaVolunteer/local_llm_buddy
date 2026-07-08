"""Streamlit application for local_llm_buddy.

Run with:
    streamlit run app.py

Environment variables (or .env file):
    PINECONE_API_KEY       – required
    PINECONE_INDEX_NAME    – default: otter-transcripts
    PINECONE_ENVIRONMENT   – default: us-east-1-aws
    OLLAMA_BASE_URL        – default: http://localhost:11434
    OLLAMA_MODEL           – default: llama3
    OLLAMA_EMBED_MODEL     – default: nomic-embed-text
    EMBED_DIMENSION        – default: 768
    CHUNK_SIZE             – default: 1000
    CHUNK_OVERLAP          – default: 200
    RETRIEVER_K            – default: 4
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from local_llm_buddy import OtterTranscriptLoader, PineconeStore, Settings, build_rag_chain

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Local LLM Buddy",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Local LLM Buddy")
st.caption(
    "Upload Otter.ai transcripts, index them in Pinecone, then chat with your"
    " local Ollama model using Retrieval-Augmented Generation."
)

# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------


def _get_settings() -> Settings:
    if "settings" not in st.session_state:
        st.session_state["settings"] = Settings()
    return st.session_state["settings"]


def _get_store() -> PineconeStore | None:
    return st.session_state.get("store")


def _get_chain():
    return st.session_state.get("chain")


# ---------------------------------------------------------------------------
# Sidebar – configuration & ingestion
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    cfg = _get_settings()

    pinecone_key = st.text_input(
        "Pinecone API Key",
        value=cfg.pinecone_api_key,
        type="password",
        help="Your Pinecone API key.",
    )
    index_name = st.text_input(
        "Pinecone Index Name",
        value=cfg.pinecone_index_name,
    )
    pinecone_env = st.text_input(
        "Pinecone Environment",
        value=cfg.pinecone_environment,
        help='e.g. "us-east-1-aws"',
    )
    ollama_url = st.text_input(
        "Ollama Base URL",
        value=cfg.ollama_base_url,
    )
    ollama_model = st.text_input(
        "Ollama LLM Model",
        value=cfg.ollama_model,
        help='e.g. "llama3", "mistral"',
    )
    embed_model = st.text_input(
        "Ollama Embed Model",
        value=cfg.ollama_embed_model,
        help='e.g. "nomic-embed-text"',
    )
    embed_dim = st.number_input(
        "Embedding Dimension",
        value=cfg.embed_dimension,
        min_value=64,
        step=64,
    )
    chunk_size = st.number_input(
        "Chunk Size (chars)",
        value=cfg.chunk_size,
        min_value=100,
        step=100,
    )
    chunk_overlap = st.number_input(
        "Chunk Overlap (chars)",
        value=cfg.chunk_overlap,
        min_value=0,
        step=50,
    )
    retriever_k = st.number_input(
        "Retrieved Chunks (k)",
        value=cfg.retriever_k,
        min_value=1,
        step=1,
    )

    if st.button("💾 Apply Settings"):
        cfg.pinecone_api_key = pinecone_key
        cfg.pinecone_index_name = index_name
        cfg.pinecone_environment = pinecone_env
        cfg.ollama_base_url = ollama_url
        cfg.ollama_model = ollama_model
        cfg.ollama_embed_model = embed_model
        cfg.embed_dimension = int(embed_dim)
        cfg.chunk_size = int(chunk_size)
        cfg.chunk_overlap = int(chunk_overlap)
        cfg.retriever_k = int(retriever_k)
        # Clear cached store / chain so they rebuild with new settings
        st.session_state.pop("store", None)
        st.session_state.pop("chain", None)
        st.success("Settings applied.")

    st.divider()
    st.header("📄 Upload Transcripts")

    uploaded_files = st.file_uploader(
        "Choose Otter.ai transcript files (.txt or .srt)",
        accept_multiple_files=True,
        type=["txt", "srt"],
    )

    if st.button("🚀 Ingest Transcripts", disabled=not uploaded_files):
        if not cfg.pinecone_api_key:
            st.error("Please enter your Pinecone API key and apply settings first.")
        else:
            with st.spinner("Loading and indexing transcripts …"):
                try:
                    # Write uploaded files to a temp directory so the loader
                    # can read them from disk.
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        paths: list[Path] = []
                        for uf in uploaded_files:
                            dest = Path(tmp_dir) / uf.name
                            dest.write_bytes(uf.read())
                            paths.append(dest)

                        loader = OtterTranscriptLoader(paths)
                        docs = loader.load()

                    store = PineconeStore(cfg)
                    n_chunks = store.ingest(docs)

                    st.session_state["store"] = store
                    st.session_state["chain"] = build_rag_chain(
                        store.retriever(), cfg
                    )
                    st.session_state.setdefault("messages", [])

                    st.success(
                        f"✅ Ingested {len(docs)} turns → {n_chunks} chunks."
                    )
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

    st.divider()
    st.header("🔗 Connect to Existing Index")

    if st.button("Connect"):
        if not cfg.pinecone_api_key:
            st.error("Please enter your Pinecone API key and apply settings first.")
        else:
            with st.spinner("Connecting to Pinecone …"):
                try:
                    store = PineconeStore(cfg)
                    st.session_state["store"] = store
                    st.session_state["chain"] = build_rag_chain(
                        store.retriever(), cfg
                    )
                    st.session_state.setdefault("messages", [])
                    st.success("Connected.")
                except Exception as exc:
                    st.error(f"Connection failed: {exc}")

# ---------------------------------------------------------------------------
# Main – chat interface
# ---------------------------------------------------------------------------

chain = _get_chain()

if chain is None:
    st.info(
        "👈 Upload transcripts and click **Ingest Transcripts**, or connect"
        " to an existing Pinecone index, to start chatting."
    )
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render existing messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# New user input
if prompt := st.chat_input("Ask a question about your transcripts …"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            try:
                # Prior turns (excluding the question just appended above)
                # so the chain can resolve follow-ups like "was that in the
                # latest meeting?" and the retriever can search accordingly.
                chat_history = []
                for msg in st.session_state["messages"][:-1]:
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(msg["content"]))
                    else:
                        chat_history.append(AIMessage(msg["content"]))
                answer = chain.invoke(
                    {"question": prompt, "chat_history": chat_history}
                )
            except Exception as exc:
                answer = f"⚠️ Error: {exc}"
        st.markdown(answer)

    st.session_state["messages"].append({"role": "assistant", "content": answer})
