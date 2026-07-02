"""LangChain document loader for Otter.ai transcript exports.

Otter.ai can export transcripts in several formats.  This module handles the
two most common ones:

* **Plain-text** (.txt) – lines of the form ``Speaker  HH:MM:SS`` followed by
  one or more lines of utterance text.
* **SRT subtitle** (.srt) – standard SRT blocks (index / timecode / text).

For every distinct speaker turn (or SRT block) a separate
:class:`~langchain_core.documents.Document` is produced, with metadata fields:

* ``source`` – absolute path to the source file.
* ``speaker`` – speaker label (empty string for SRT).
* ``timestamp`` – start timestamp string.
* ``format`` – ``"txt"`` or ``"srt"``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, List

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Matches lines like:  "John Smith  00:01:23" or "SPEAKER_01  1:23:45"
_TXT_HEADER_RE = re.compile(
    r"^(?P<speaker>.+?)\s{2,}(?P<timestamp>\d{1,2}:\d{2}(?::\d{2})?)\s*$"
)

# Matches SRT timecode lines:  "00:00:05,000 --> 00:00:10,000"
_SRT_TIMECODE_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}\s*$"
)


def _parse_txt(text: str, source: str) -> List[Document]:
    """Parse a plain-text Otter.ai export into Documents."""
    docs: List[Document] = []
    current_speaker = ""
    current_timestamp = ""
    current_lines: List[str] = []

    def _flush() -> None:
        content = " ".join(current_lines).strip()
        if content:
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": source,
                        "speaker": current_speaker,
                        "timestamp": current_timestamp,
                        "format": "txt",
                    },
                )
            )

    for line in text.splitlines():
        m = _TXT_HEADER_RE.match(line)
        if m:
            _flush()
            current_speaker = m.group("speaker").strip()
            current_timestamp = m.group("timestamp").strip()
            current_lines = []
        else:
            stripped = line.strip()
            if stripped:
                current_lines.append(stripped)

    _flush()
    return docs


def _parse_srt(text: str, source: str) -> List[Document]:
    """Parse an SRT subtitle file into Documents."""
    docs: List[Document] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        # Skip blank lines and index numbers
        if not lines[i].strip():
            i += 1
            continue
        if lines[i].strip().isdigit():
            i += 1
            continue
        # Timecode line?
        if _SRT_TIMECODE_RE.match(lines[i]):
            timestamp = lines[i].split("-->")[0].strip()
            i += 1
            content_lines: List[str] = []
            while i < len(lines) and lines[i].strip():
                content_lines.append(lines[i].strip())
                i += 1
            content = " ".join(content_lines).strip()
            if content:
                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": source,
                            "speaker": "",
                            "timestamp": timestamp,
                            "format": "srt",
                        },
                    )
                )
        else:
            i += 1
    return docs


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class OtterTranscriptLoader(BaseLoader):
    """Load one or more Otter.ai transcript files.

    Parameters
    ----------
    file_paths:
        A single path string/Path, or a list of paths, pointing to Otter.ai
        export files (.txt or .srt).  Glob patterns are **not** expanded here;
        pass a pre-expanded list when needed.
    encoding:
        Text encoding to use when reading files (default: ``"utf-8"``).

    Examples
    --------
    Load a single transcript::

        from local_llm_buddy import OtterTranscriptLoader

        loader = OtterTranscriptLoader("meeting_2024-01-15.txt")
        docs = loader.load()

    Load multiple transcripts::

        loader = OtterTranscriptLoader([
            "meeting_a.txt",
            "meeting_b.srt",
        ])
        docs = loader.load()
    """

    def __init__(
        self,
        file_paths: "str | Path | List[str | Path]",
        encoding: str = "utf-8",
    ) -> None:
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        self._paths: List[Path] = [Path(p) for p in file_paths]
        self._encoding = encoding

    # ------------------------------------------------------------------
    # BaseLoader interface
    # ------------------------------------------------------------------

    def lazy_load(self) -> Iterator[Document]:
        """Yield Documents one by one (memory-efficient for large corpora)."""
        for path in self._paths:
            yield from self._load_file(path)

    def load(self) -> List[Document]:
        """Load all documents and return them as a list."""
        return list(self.lazy_load())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_file(self, path: Path) -> List[Document]:
        if not path.exists():
            raise FileNotFoundError(f"Transcript file not found: {path}")
        text = path.read_text(encoding=self._encoding)
        suffix = path.suffix.lower()
        source = str(path.resolve())
        if suffix == ".srt":
            return _parse_srt(text, source)
        # Default: treat as plain-text Otter.ai export
        docs = _parse_txt(text, source)
        if not docs:
            # Fallback: wrap the entire file as one document
            docs = [
                Document(
                    page_content=text.strip(),
                    metadata={
                        "source": source,
                        "speaker": "",
                        "timestamp": "",
                        "format": "txt",
                    },
                )
            ]
        return docs
