"""Manual smoke test: load the sample transcript, chunk it, embed it via
Ollama, and upsert into Pinecone. Then run a similarity search to confirm
the vectors are retrievable.

Run with:
    python test.py
"""

from local_llm_buddy import OtterTranscriptLoader, PineconeStore, Settings

TRANSCRIPT_PATH = "transripts/GC Dev chat_otter_ai_transcript.txt"


def main() -> None:
    settings = Settings()
    settings.validate()

    loader = OtterTranscriptLoader(TRANSCRIPT_PATH)
    docs = loader.load()
    print(f"Loaded {len(docs)} turns from {TRANSCRIPT_PATH}")
    print(f"Sample doc: {docs[0]}")

    store = PineconeStore(settings)
    n_chunks = store.ingest(docs)
    print(f"Ingested {n_chunks} chunks into Pinecone index '{settings.pinecone_index_name}'")

    results = store.vectorstore.similarity_search("What did they discuss about HVAC?", k=3)
    print(f"\nSimilarity search returned {len(results)} results:")
    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Speaker: {r.metadata.get('speaker')}  Timestamp: {r.metadata.get('timestamp')}")
        print(r.page_content[:200])


if __name__ == "__main__":
    main()


