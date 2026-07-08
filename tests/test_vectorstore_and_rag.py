"""Tests for PineconeStore and build_rag_chain using mocked external clients.

No real Pinecone or Ollama connection is made – all network calls are patched.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama

from local_llm_buddy.config import Settings
from local_llm_buddy.loader import OtterTranscriptLoader

TXT_TRANSCRIPT = textwrap.dedent(
    """\
    Alice  00:00:05
    Hello everyone, welcome to the Q3 planning meeting.

    Bob  00:00:12
    Thanks Alice. Let me share my screen.
    """
)


def _make_settings(**overrides) -> Settings:
    cfg = Settings.__new__(Settings)
    cfg.pinecone_api_key = "pk-test"
    cfg.pinecone_index_name = "test-index"
    cfg.pinecone_environment = "us-east-1-aws"
    cfg.ollama_base_url = "http://localhost:11434"
    cfg.ollama_model = "llama3"
    cfg.ollama_embed_model = "nomic-embed-text"
    cfg.embed_dimension = 768
    cfg.chunk_size = 500
    cfg.chunk_overlap = 50
    cfg.retriever_k = 2
    cfg.llm_temperature = 0.0
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# PineconeStore tests (mocked)
# ---------------------------------------------------------------------------


class TestPineconeStore:
    """Tests for PineconeStore that mock all external I/O."""

    def _build_store(self, existing_indexes: List[str] | None = None):
        """Return a PineconeStore with all external dependencies mocked."""
        from local_llm_buddy.vectorstore import PineconeStore

        cfg = _make_settings()
        existing_indexes = existing_indexes or []

        # Mock Pinecone client
        mock_pc = MagicMock()
        mock_index_info = MagicMock()
        mock_index_info.name = cfg.pinecone_index_name
        mock_pc.list_indexes.return_value = (
            [mock_index_info] if cfg.pinecone_index_name in existing_indexes else []
        )

        # Mock embeddings
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1] * 768]
        mock_embeddings.embed_query.return_value = [0.1] * 768

        # Mock vector store
        mock_vs = MagicMock()
        mock_vs.add_documents.return_value = None
        mock_vs.as_retriever.return_value = MagicMock()

        with (
            patch("local_llm_buddy.vectorstore.Pinecone", return_value=mock_pc),
            patch(
                "local_llm_buddy.vectorstore.OllamaEmbeddings",
                return_value=mock_embeddings,
            ),
            patch(
                "local_llm_buddy.vectorstore.PineconeVectorStore",
                return_value=mock_vs,
            ),
        ):
            store = PineconeStore(cfg)

        # Expose mocks for assertions
        store._mock_pc = mock_pc
        store._mock_vs = mock_vs
        return store

    def test_creates_index_when_missing(self):
        store = self._build_store(existing_indexes=[])
        store._mock_pc.create_index.assert_called_once()

    def test_skips_create_when_index_exists(self):
        store = self._build_store(existing_indexes=["test-index"])
        store._mock_pc.create_index.assert_not_called()

    def test_ingest_calls_add_documents(self, tmp_path):
        f = tmp_path / "meeting.txt"
        f.write_text(TXT_TRANSCRIPT, encoding="utf-8")
        docs = OtterTranscriptLoader(f).load()

        store = self._build_store()
        n = store.ingest(docs)

        store._mock_vs.add_documents.assert_called_once()
        assert n >= 1

    def test_ingest_returns_chunk_count(self, tmp_path):
        f = tmp_path / "meeting.txt"
        f.write_text(TXT_TRANSCRIPT, encoding="utf-8")
        docs = OtterTranscriptLoader(f).load()

        store = self._build_store()
        n = store.ingest(docs)
        assert isinstance(n, int)
        assert n > 0

    def test_ingest_empty_docs_returns_zero(self):
        store = self._build_store()
        n = store.ingest([])
        assert n == 0
        store._mock_vs.add_documents.assert_not_called()

    def test_retriever_calls_as_retriever(self):
        store = self._build_store()
        retriever = store.retriever()
        store._mock_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 2})
        assert retriever is not None

    def test_retriever_custom_k(self):
        store = self._build_store()
        store.retriever(k=10)
        store._mock_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 10})


# ---------------------------------------------------------------------------
# build_rag_chain tests (mocked)
# ---------------------------------------------------------------------------


class TestBuildRagChain:
    def test_chain_is_callable(self):
        from local_llm_buddy.rag import build_rag_chain

        cfg = _make_settings()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []

        with patch("local_llm_buddy.rag.ChatOllama") as MockLLM:
            mock_llm_instance = MagicMock()
            mock_llm_instance.invoke.return_value = MagicMock(content="test answer")
            MockLLM.return_value = mock_llm_instance
            chain = build_rag_chain(mock_retriever, cfg)

        assert callable(chain.invoke)

    def test_llm_constructed_with_settings(self):
        from local_llm_buddy.rag import build_rag_chain

        cfg = _make_settings(ollama_model="mistral", llm_temperature=0.7)
        mock_retriever = MagicMock()

        with patch("local_llm_buddy.rag.ChatOllama") as MockLLM:
            build_rag_chain(mock_retriever, cfg)
            MockLLM.assert_called_once_with(
                base_url=cfg.ollama_base_url,
                model="mistral",
                temperature=0.7,
            )

    def test_format_docs_includes_speaker(self):
        from local_llm_buddy.rag import _format_docs

        docs = [
            Document(
                page_content="Hello world",
                metadata={"speaker": "Alice", "timestamp": "00:01", "format": "txt"},
            )
        ]
        result = _format_docs(docs)
        assert "Alice" in result
        assert "Hello world" in result

    def test_format_docs_no_speaker(self):
        from local_llm_buddy.rag import _format_docs

        docs = [
            Document(
                page_content="Some content",
                metadata={"speaker": "", "timestamp": "00:01", "format": "srt"},
            )
        ]
        result = _format_docs(docs)
        assert "Some content" in result

    def test_format_docs_multiple_chunks_separated(self):
        from local_llm_buddy.rag import _format_docs

        docs = [
            Document(page_content="Chunk A", metadata={"speaker": "", "timestamp": ""}),
            Document(page_content="Chunk B", metadata={"speaker": "", "timestamp": ""}),
        ]
        result = _format_docs(docs)
        assert "---" in result
        assert "Chunk A" in result
        assert "Chunk B" in result

    def test_temperature_passed_to_llm(self):
        from local_llm_buddy.rag import build_rag_chain

        cfg = _make_settings(llm_temperature=0.5)
        mock_retriever = MagicMock()

        with patch("local_llm_buddy.rag.ChatOllama") as MockLLM:
            build_rag_chain(mock_retriever, cfg)
            _, kwargs = MockLLM.call_args
            assert kwargs["temperature"] == 0.5

    def test_invoke_without_history_skips_condensing(self):
        """A bare string question (no memory) shouldn't trigger the extra
        condense-question LLM call – only the final answer call."""
        from langchain_core.runnables import RunnableLambda

        from local_llm_buddy.rag import build_rag_chain

        cfg = _make_settings()
        retriever_calls = []
        # A MagicMock(spec=BaseRetriever) would auto-mock __or__/__ror__ too,
        # silently short-circuiting the real LCEL pipe composition. Use a
        # genuine Runnable (RunnableLambda) so `retriever | fn` behaves for real.
        fake_retriever = RunnableLambda(lambda q: (retriever_calls.append(q), [])[1])

        with patch("local_llm_buddy.rag.ChatOllama") as MockLLM:
            mock_llm_instance = MagicMock(spec=ChatOllama)
            mock_llm_instance.invoke.return_value = AIMessage(content="test answer")
            MockLLM.return_value = mock_llm_instance
            chain = build_rag_chain(fake_retriever, cfg)
            answer = chain.invoke("What did Jon say about LLMs?")

        assert answer == "test answer"
        assert mock_llm_instance.invoke.call_count == 1
        assert retriever_calls == ["What did Jon say about LLMs?"]

    def test_invoke_with_history_condenses_and_retrieves_standalone_question(self):
        """A dict with chat_history should condense the follow-up into a
        standalone question before retrieval, and pass history to the LLM."""
        from langchain_core.messages import HumanMessage
        from langchain_core.runnables import RunnableLambda

        from local_llm_buddy.rag import build_rag_chain

        cfg = _make_settings()
        retriever_calls = []
        fake_retriever = RunnableLambda(lambda q: (retriever_calls.append(q), [])[1])
        chat_history = [
            HumanMessage("What did Jon say about LLMs?"),
            AIMessage("Jon said LLMs are useful for note-taking."),
        ]

        with patch("local_llm_buddy.rag.ChatOllama") as MockLLM:
            mock_llm_instance = MagicMock(spec=ChatOllama)
            mock_llm_instance.invoke.return_value = AIMessage(
                content="Was that in the latest meeting?"
            )
            MockLLM.return_value = mock_llm_instance
            chain = build_rag_chain(fake_retriever, cfg)
            answer = chain.invoke(
                {
                    "question": "Was that in the latest meeting?",
                    "chat_history": chat_history,
                }
            )

        assert answer == "Was that in the latest meeting?"
        # One call to condense the standalone question, one to answer.
        assert mock_llm_instance.invoke.call_count == 2
        assert retriever_calls == ["Was that in the latest meeting?"]
