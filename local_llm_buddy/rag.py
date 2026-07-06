"""RAG chain construction for local_llm_buddy.

The chain follows the standard Retrieval-Augmented Generation pattern:

1. A user question is passed to a Pinecone retriever.
2. Retrieved chunks are inserted into a prompt template.
3. The prompt is sent to a local Ollama model.
4. The model's answer is returned as a string.

Typical usage::

    from local_llm_buddy import Settings, PineconeStore, build_rag_chain

    settings = Settings()
    store = PineconeStore(settings)
    chain = build_rag_chain(store.retriever(), settings)

    answer = chain.invoke("What did Alice say about the Q3 roadmap?")
    print(answer)
"""

from __future__ import annotations

from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_ollama import ChatOllama


from .config import Settings

# ---------------------------------------------------------------------------
# Prompt template
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


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_rag_chain(
    retriever,
    settings: Optional[Settings] = None,
) -> RunnableSerializable:
    """Build and return a LangChain RAG chain.

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
        A chain that accepts a ``str`` question and returns a ``str`` answer.
        Call it with ``chain.invoke("your question here")``.
    """
    cfg = settings or Settings()

    llm = ChatOllama(
        base_url=cfg.ollama_base_url,
        model=cfg.ollama_model,
        temperature=cfg.llm_temperature,
    )

    chain: RunnableSerializable = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | _PROMPT
        | llm
        | StrOutputParser()
    )

    return chain
