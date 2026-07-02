"""Configuration management for local_llm_buddy.

All settings are read from environment variables (or a .env file loaded by
python-dotenv).  Sensible defaults are provided wherever possible so that the
package works out-of-the-box for local experimentation.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Central configuration object.

    Attributes
    ----------
    pinecone_api_key:
        Pinecone API key (required for cloud-hosted Pinecone).
    pinecone_index_name:
        Name of the Pinecone index to create / connect to.
    pinecone_environment:
        Pinecone environment / cloud region (e.g. ``"us-east-1-aws"``).
    ollama_base_url:
        Base URL of the running Ollama server.
    ollama_model:
        Ollama model tag to use as the LLM (e.g. ``"llama3"``).
    ollama_embed_model:
        Ollama model tag to use for embeddings (e.g. ``"nomic-embed-text"``).
    embed_dimension:
        Dimension of the embedding vectors produced by ``ollama_embed_model``.
        Must match the dimension configured on the Pinecone index.
    chunk_size:
        Number of characters per text chunk when splitting transcripts.
    chunk_overlap:
        Number of characters of overlap between consecutive chunks.
    retriever_k:
        Number of chunks to retrieve from Pinecone per query.
    llm_temperature:
        Sampling temperature for the Ollama LLM (0 = deterministic,
        higher = more creative).  Defaults to ``0``.
    """

    def __init__(self) -> None:
        # Pinecone
        self.pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
        self.pinecone_index_name: str = os.getenv(
            "PINECONE_INDEX_NAME", "otter-transcripts"
        )
        self.pinecone_environment: str = os.getenv(
            "PINECONE_ENVIRONMENT", "us-east-1-aws"
        )

        # Ollama
        self.ollama_base_url: str = os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3")
        self.ollama_embed_model: str = os.getenv(
            "OLLAMA_EMBED_MODEL", "nomic-embed-text"
        )
        self.embed_dimension: int = int(os.getenv("EMBED_DIMENSION", "768"))

        # Chunking
        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

        # Retrieval
        self.retriever_k: int = int(os.getenv("RETRIEVER_K", "4"))

        # LLM generation
        self.llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))

    def validate(self) -> None:
        """Raise ``ValueError`` if required settings are missing."""
        if not self.pinecone_api_key:
            raise ValueError(
                "PINECONE_API_KEY environment variable is not set. "
                "Please add it to your .env file or shell environment."
            )
