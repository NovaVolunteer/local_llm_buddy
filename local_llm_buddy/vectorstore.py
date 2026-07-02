"""Pinecone vector store helpers.

This module wraps the LangChain + Pinecone integration so that callers can
focus on data rather than plumbing.

Typical usage::

    from local_llm_buddy import Settings, OtterTranscriptLoader, PineconeStore

    settings = Settings()
    loader = OtterTranscriptLoader("meeting.txt")
    docs = loader.load()

    store = PineconeStore(settings)
    store.ingest(docs)          # split, embed and upsert into Pinecone
    retriever = store.retriever()  # LangChain retriever ready for RAG
"""

from __future__ import annotations

from typing import List, Optional

from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from .config import Settings


class PineconeStore:
    """Manage a Pinecone index for Otter.ai transcript chunks.

    Parameters
    ----------
    settings:
        A :class:`~local_llm_buddy.config.Settings` instance.  When omitted a
        new instance is created from environment variables.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._cfg = settings or Settings()
        self._cfg.validate()

        # Ollama embeddings (local, no API key needed)
        self._embeddings = OllamaEmbeddings(
            base_url=self._cfg.ollama_base_url,
            model=self._cfg.ollama_embed_model,
        )

        # Text splitter for chunking transcripts
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._cfg.chunk_size,
            chunk_overlap=self._cfg.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

        # Pinecone client
        self._pc = Pinecone(api_key=self._cfg.pinecone_api_key)
        self._ensure_index()

        # LangChain vector store wrapper
        self._vectorstore = PineconeVectorStore(
            index_name=self._cfg.pinecone_index_name,
            embedding=self._embeddings,
            pinecone_api_key=self._cfg.pinecone_api_key,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, docs: List[Document]) -> int:
        """Split *docs* into chunks and upsert them into Pinecone.

        Parameters
        ----------
        docs:
            Raw :class:`~langchain_core.documents.Document` objects (e.g. from
            :class:`~local_llm_buddy.loader.OtterTranscriptLoader`).

        Returns
        -------
        int
            Number of chunks upserted.
        """
        chunks = self._splitter.split_documents(docs)
        if not chunks:
            return 0
        self._vectorstore.add_documents(chunks)
        return len(chunks)

    def retriever(self, k: Optional[int] = None):
        """Return a LangChain retriever backed by the Pinecone index.

        Parameters
        ----------
        k:
            Number of chunks to retrieve per query.  Defaults to
            ``settings.retriever_k``.
        """
        k = k if k is not None else self._cfg.retriever_k
        return self._vectorstore.as_retriever(search_kwargs={"k": k})

    @property
    def vectorstore(self) -> PineconeVectorStore:
        """Direct access to the underlying LangChain ``PineconeVectorStore``."""
        return self._vectorstore

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_index(self) -> None:
        """Create the Pinecone index if it does not already exist."""
        existing = [idx.name for idx in self._pc.list_indexes()]
        if self._cfg.pinecone_index_name not in existing:
            self._pc.create_index(
                name=self._cfg.pinecone_index_name,
                dimension=self._cfg.embed_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=self._cfg.pinecone_environment,
                ),
            )
