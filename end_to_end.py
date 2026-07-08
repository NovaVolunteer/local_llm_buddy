# %%
#Need to create script that runs the entire 
# end to end process of loading a transcript, chunking it, 
# embedding it, and upserting into Pinecone. 
# Then build the RA (Retrieval-Augmented) model and query the results.

from ingest import ingest_files
from local_llm_buddy import build_rag_chain, Settings, PineconeStore, OtterTranscriptLoader
from langchain_core.messages import AIMessage, HumanMessage
# %%
settings = Settings()
store = PineconeStore(settings)
# %%
ingest_files(transcripts=["transripts\ATA Zero Day Form Extraction Meeting_6.5.26.txt",
                          "transripts\Bi-Weekly ATA_6.26.26.txt"],
             summaries=["transripts\ATA_summary_6.26.26.txt",
                        "transripts\ATA_zero_Day_summary6.5.26.txt",
                        "transripts\old_note_summary.txt"])


# %%
chain = build_rag_chain(store.retriever(), settings)

def ask(question: str) -> str:
    """Ask a question, keeping it in the running chat_history for follow-ups."""
    answer = chain.invoke({"question": question, "chat_history": chat_history})
    chat_history.append(HumanMessage(question))
    chat_history.append(AIMessage(answer))
    print(answer)
    return answer

# %%
# Re-run THIS cell to start a fresh conversation (e.g. before re-running an
# ask() cell you already ran, or after making changes) — chat_history keeps
# accumulating across reruns otherwise, which bloats context until the model
# starts repeating itself regardless of the question.
chat_history = []

# %%
ask("What did Jon say about LLMs?")

# %%
ask("What did Van discuss at the last meeting?")

# %%
ask("What did Jonathan discuss in the meeting on 6.26.26")
# %%
ask("give men a summary of the meeting on 6.26.26")
# %%
