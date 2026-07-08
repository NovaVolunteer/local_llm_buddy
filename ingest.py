"""Ingest Otter.ai transcript and/or meeting summary files into Pinecone.

Run from the terminal:
    python ingest.py --transcripts "transripts/Meeting.txt" --summaries "transripts/Meeting_summary.txt"

Or call from a script/notebook:
    from ingest import ingest_files
    ingest_files(transcripts=["transripts/Meeting.txt"], summaries=["transripts/Meeting_summary.txt"])

Files passed as transcripts are tagged doc_type="transcript"; files passed as
summaries are tagged doc_type="summary". At least one of the two must be given.

Only pass files that haven't been ingested yet — re-ingesting the same file
will add duplicate chunks, since there's no de-duplication against what's
already in the index.
"""

import argparse
from pathlib import Path
from typing import Optional, Sequence, Union

from local_llm_buddy import OtterTranscriptLoader, PineconeStore, Settings


def ingest_files(
    transcripts: Optional[Sequence[Union[str, Path]]] = None,
    summaries: Optional[Sequence[Union[str, Path]]] = None,
    settings: Optional[Settings] = None,
) -> int:
    """Load and ingest transcript/summary files into Pinecone.

    Returns the total number of chunks ingested.
    """
    transcripts = transcripts or []
    summaries = summaries or []
    if not transcripts and not summaries:
        raise ValueError("pass at least one file via transcripts and/or summaries")

    settings = settings or Settings()
    settings.validate()
    store = PineconeStore(settings)

    total_chunks = 0
    if transcripts:
        docs = OtterTranscriptLoader(transcripts, doc_type="transcript").load()
        print(f"Loaded {len(docs)} turns from {len(transcripts)} transcript file(s)")
        n = store.ingest(docs)
        print(f"Ingested {n} transcript chunks")
        total_chunks += n

    if summaries:
        docs = OtterTranscriptLoader(summaries, doc_type="summary").load()
        print(f"Loaded {len(docs)} document(s) from {len(summaries)} summary file(s)")
        n = store.ingest(docs)
        print(f"Ingested {n} summary chunks")
        total_chunks += n

    print(f"Total: {total_chunks} chunks ingested into Pinecone index '{settings.pinecone_index_name}'")
    return total_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transcripts", nargs="+", default=[], help="Raw transcript file(s)"
    )
    parser.add_argument(
        "--summaries", nargs="+", default=[], help="Meeting summary file(s)"
    )
    args = parser.parse_args()

    if not args.transcripts and not args.summaries:
        parser.error("pass at least one file via --transcripts and/or --summaries")

    ingest_files(transcripts=args.transcripts, summaries=args.summaries)


if __name__ == "__main__":
    main()