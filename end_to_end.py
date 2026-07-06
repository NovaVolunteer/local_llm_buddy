# %%
#Need to create script that runs the entire 
# end to end process of loading a transcript, chunking it, 
# embedding it, and upserting into Pinecone. 
# Then build the RA (Retrieval-Augmented) model and query the results.

from local_llm_buddy import build_rag_chain, Settings, PineconeStore, OtterTranscriptLoader

# %%
settings = Settings()
store = PineconeStore(settings)
# %%
chain = build_rag_chain(store.retriever(), settings)
answer = chain.invoke("What did Jon say about LLMs?")
print(answer)

# %%
answer = chain.invoke("What did Van discuss at the last meeting?")
print(answer)

# %%
answer = chain.invoke("What did Jonathan discuss in the meeting on 6.26.26")
print(answer)
# %%
