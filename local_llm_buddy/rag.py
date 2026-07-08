"""RAG chain construction for local_llm_buddy.

The chain follows a conversational Retrieval-Augmented Generation pattern:

1. If chat history is present, the question is first rewritten into a
   standalone question (so follow-ups like "was that in the latest meeting?"
   resolve correctly against the vector store).
2. The standalone question is passed to a Pinecone retriever.
3. Retrieved chunks + the original chat history are inserted into a prompt
   template.
4. The prompt is sent to a local Ollama model.
5. The model's answer is returned as a string.

Typical usage::

    from local_llm_buddy import Settings, PineconeStore, build_rag_chain

    settings = Settings()
    store = PineconeStore(settings)
    chain = build_rag_chain(store.retriever(), settings)

    # First turn – no history needed.
    answer = chain.invoke("What did Alice say about the Q3 roadmap?")

    # Follow-up turn – pass the running chat history so the model (and the
    # retriever) have the context of the conversation so far.
    chat_history = [
        HumanMessage("What did Alice say about the Q3 roadmap?"),
        AIMessage(answer),
    ]
    answer = chain.invoke(
        {"question": "Was that in the latest meeting?", "chat_history": chat_history}
    )
"""

from __future__ import annotations

from typing import Optional, Union

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableSerializable
from langchain_ollama import ChatOllama


from .config import Settings

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on meeting "
    "transcripts. Use only the provided context to answer. If the answer "
    "cannot be found in the context, say so honestly.\n\n"
    "Context:\n{context}"
)

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

_CONDENSE_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question which might reference "
    "context in the chat history, rewrite it into a standalone question that "
    "can be understood without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as-is."
)

_CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _CONDENSE_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_docs(docs) -> str:
    """Concatenate retrieved document chunks into a single context string."""
    parts = []
    for doc in docs:
        meta = doc.metadata
        header_parts = []
        if meta.get("speaker"):
            header_parts.append(f"Speaker: {meta['speaker']}")
        if meta.get("timestamp"):
            header_parts.append(f"[{meta['timestamp']}]")
        header = "  ".join(header_parts)
        if header:
            parts.append(f"{header}\n{doc.page_content}")
        else:
            parts.append(doc.page_content)
    return "\n\n---\n\n".join(parts)


def _normalize_input(value: Union[str, dict]) -> dict:
    """Accept either a bare question string or a dict with chat history."""
    if isinstance(value, str):
        return {"question": value, "chat_history": []}
    return {
        "question": value["question"],
        "chat_history": value.get("chat_history", []),
    }


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_rag_chain(
    retriever,
    settings: Optional[Settings] = None,
) -> RunnableSerializable:
    """Build and return a conversational LangChain RAG chain.

    Parameters
    ----------
    retriever:
        Any LangChain retriever (e.g. from
        :meth:`~local_llm_buddy.vectorstore.PineconeStore.retriever`).
    settings:
        Configuration object.  When omitted a new :class:`Settings` instance
        is created from environment variables.

    Returns
    -------
    RunnableSerializable
        A chain that returns a ``str`` answer. Call it with either:

        * ``chain.invoke("your question here")`` – no memory, single-turn.
        * ``chain.invoke({"question": "...", "chat_history": [...]})`` –
          ``chat_history`` is a list of :class:`~langchain_core.messages.BaseMessage`
          (e.g. alternating :class:`HumanMessage`/:class:`AIMessage`) from
          earlier turns in the conversation.
    """
    cfg = settings or Settings()

    llm = ChatOllama(
        base_url=cfg.ollama_base_url,
        model=cfg.ollama_model,
        temperature=cfg.llm_temperature,
    )

    condense_chain = _CONDENSE_PROMPT | llm | StrOutputParser()

    def _standalone_question(inputs: dict) -> str:
        if not inputs["chat_history"]:
            return inputs["question"]
        return condense_chain.invoke(inputs)

    chain: RunnableSerializable = (
        RunnableLambda(_normalize_input)
        | RunnablePassthrough.assign(standalone_question=RunnableLambda(_standalone_question))
        | RunnablePassthrough.assign(
            context=(lambda x: x["standalone_question"]) | retriever | _format_docs
        )
        | _PROMPT
        | llm
        | StrOutputParser()
    )

    return chain
