"""Tests for local_llm_buddy – no external services required.

These tests cover pure-Python logic (transcript parsing, configuration) and
use lightweight stubs for any component that would require Pinecone or Ollama.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local_llm_buddy.config import Settings
from local_llm_buddy.loader import OtterTranscriptLoader, _parse_srt, _parse_txt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TXT_TRANSCRIPT = textwrap.dedent(
    """\
    Alice  00:00:05
    Hello everyone, welcome to the Q3 planning meeting.

    Bob  00:00:12
    Thanks Alice. Let me share my screen.

    Alice  00:00:18
    Sure, go ahead. We'll start with the roadmap.
    """
)

SRT_TRANSCRIPT = textwrap.dedent(
    """\
    1
    00:00:05,000 --> 00:00:09,000
    Hello everyone, welcome to the Q3 planning meeting.

    2
    00:00:10,000 --> 00:00:14,000
    Thanks Alice. Let me share my screen.

    3
    00:00:15,000 --> 00:00:20,000
    Sure, go ahead. We'll start with the roadmap.
    """
)


# ---------------------------------------------------------------------------
# Settings tests
# ---------------------------------------------------------------------------


class TestSettings:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)
        monkeypatch.delenv("PINECONE_INDEX_NAME", raising=False)
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        cfg = Settings()
        assert cfg.pinecone_index_name == "otter-transcripts"
        assert cfg.ollama_model == "llama3"
        assert cfg.chunk_size == 1000
        assert cfg.chunk_overlap == 200
        assert cfg.retriever_k == 4

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("PINECONE_API_KEY", "pk-test")
        monkeypatch.setenv("OLLAMA_MODEL", "mistral")
        monkeypatch.setenv("CHUNK_SIZE", "500")
        cfg = Settings()
        assert cfg.pinecone_api_key == "pk-test"
        assert cfg.ollama_model == "mistral"
        assert cfg.chunk_size == 500

    def test_validate_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)
        cfg = Settings()
        cfg.pinecone_api_key = ""
        with pytest.raises(ValueError, match="PINECONE_API_KEY"):
            cfg.validate()

    def test_validate_passes_with_key(self, monkeypatch):
        monkeypatch.setenv("PINECONE_API_KEY", "pk-test")
        cfg = Settings()
        cfg.validate()  # should not raise


# ---------------------------------------------------------------------------
# Loader – plain text parsing
# ---------------------------------------------------------------------------


class TestParseTxt:
    def test_speaker_count(self):
        docs = _parse_txt(TXT_TRANSCRIPT, "fake.txt")
        assert len(docs) == 3

    def test_speaker_labels(self):
        docs = _parse_txt(TXT_TRANSCRIPT, "fake.txt")
        speakers = [d.metadata["speaker"] for d in docs]
        assert speakers == ["Alice", "Bob", "Alice"]

    def test_timestamps(self):
        docs = _parse_txt(TXT_TRANSCRIPT, "fake.txt")
        assert docs[0].metadata["timestamp"] == "00:00:05"
        assert docs[1].metadata["timestamp"] == "00:00:12"

    def test_content(self):
        docs = _parse_txt(TXT_TRANSCRIPT, "fake.txt")
        assert "Q3 planning" in docs[0].page_content

    def test_metadata_format(self):
        docs = _parse_txt(TXT_TRANSCRIPT, "fake.txt")
        for d in docs:
            assert d.metadata["format"] == "txt"
            assert d.metadata["source"] == "fake.txt"

    def test_empty_input(self):
        docs = _parse_txt("", "empty.txt")
        assert docs == []


# ---------------------------------------------------------------------------
# Loader – SRT parsing
# ---------------------------------------------------------------------------


class TestParseSrt:
    def test_block_count(self):
        docs = _parse_srt(SRT_TRANSCRIPT, "fake.srt")
        assert len(docs) == 3

    def test_content(self):
        docs = _parse_srt(SRT_TRANSCRIPT, "fake.srt")
        assert "Q3 planning" in docs[0].page_content

    def test_speaker_empty_for_srt(self):
        docs = _parse_srt(SRT_TRANSCRIPT, "fake.srt")
        assert all(d.metadata["speaker"] == "" for d in docs)

    def test_metadata_format(self):
        docs = _parse_srt(SRT_TRANSCRIPT, "fake.srt")
        for d in docs:
            assert d.metadata["format"] == "srt"

    def test_timestamp_extracted(self):
        docs = _parse_srt(SRT_TRANSCRIPT, "fake.srt")
        assert "00:00:05,000" in docs[0].metadata["timestamp"]

    def test_empty_srt(self):
        docs = _parse_srt("", "empty.srt")
        assert docs == []


# ---------------------------------------------------------------------------
# OtterTranscriptLoader – file I/O
# ---------------------------------------------------------------------------


class TestOtterTranscriptLoader:
    def test_load_txt_file(self, tmp_path):
        f = tmp_path / "meeting.txt"
        f.write_text(TXT_TRANSCRIPT, encoding="utf-8")
        loader = OtterTranscriptLoader(f)
        docs = loader.load()
        assert len(docs) == 3

    def test_load_srt_file(self, tmp_path):
        f = tmp_path / "meeting.srt"
        f.write_text(SRT_TRANSCRIPT, encoding="utf-8")
        loader = OtterTranscriptLoader(f)
        docs = loader.load()
        assert len(docs) == 3

    def test_load_multiple_files(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.srt"
        f1.write_text(TXT_TRANSCRIPT, encoding="utf-8")
        f2.write_text(SRT_TRANSCRIPT, encoding="utf-8")
        loader = OtterTranscriptLoader([f1, f2])
        docs = loader.load()
        assert len(docs) == 6

    def test_lazy_load_is_iterator(self, tmp_path):
        f = tmp_path / "meeting.txt"
        f.write_text(TXT_TRANSCRIPT, encoding="utf-8")
        loader = OtterTranscriptLoader(f)
        it = loader.lazy_load()
        first = next(it)
        assert first.page_content  # non-empty

    def test_missing_file_raises(self, tmp_path):
        loader = OtterTranscriptLoader(tmp_path / "nonexistent.txt")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_fallback_for_unstructured_txt(self, tmp_path):
        """A .txt file with no speaker headers should be wrapped as one doc."""
        f = tmp_path / "plain.txt"
        f.write_text("Just some plain text without headers.", encoding="utf-8")
        loader = OtterTranscriptLoader(f)
        docs = loader.load()
        assert len(docs) == 1
        assert "plain text" in docs[0].page_content

    def test_source_is_absolute_path(self, tmp_path):
        f = tmp_path / "meeting.txt"
        f.write_text(TXT_TRANSCRIPT, encoding="utf-8")
        loader = OtterTranscriptLoader(f)
        docs = loader.load()
        for doc in docs:
            assert Path(doc.metadata["source"]).is_absolute()
