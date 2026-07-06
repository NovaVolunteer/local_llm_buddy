"""Ingest one or more Otter.ai transcript files into Pinecone.

Run with:
    python ingest.py "transripts/My Meeting.txt" "transripts/Another One.txt"

Only pass files that haven't been ingested yet — re-ingesting the same file
will add duplicate chunks, since there's no de-duplication against what's
already in the index.
"""

import sys

from local_llm_buddy import OtterTranscriptLoader, PineconeStore, Settings


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <file1> [file2 ...]")
        raise SystemExit(1)

    paths = sys.argv[1:]
    settings = Settings()
    settings.validate()

    loader = OtterTranscriptLoader(paths)
    docs = loader.load()
    print(f"Loaded {len(docs)} turns from {len(paths)} file(s)")

    store = PineconeStore(settings)
    n_chunks = store.ingest(docs)
    print(f"Ingested {n_chunks} chunks into Pinecone index '{settings.pinecone_index_name}'")


if __name__ == "__main__":
    main()
