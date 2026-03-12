from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

from app.services.llm_service import build_qa_runnable, FALLBACK_MESSAGE, NON_HEALTHCARE_MESSAGE, GREETING_MESSAGE
from app.rag.retriever import retrieve_relevant_documents

_SESSION_HISTORIES: Dict[str, InMemoryChatMessageHistory] = {}
_MAX_HISTORY_MESSAGES = 10


def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    history = _SESSION_HISTORIES.get(session_id)
    if history is None:
        history = InMemoryChatMessageHistory()
        _SESSION_HISTORIES[session_id] = history
    if len(history.messages) > _MAX_HISTORY_MESSAGES:
        history.messages = history.messages[-_MAX_HISTORY_MESSAGES:]
    return history


def _trim_session_history(session_id: str) -> None:
    history = _SESSION_HISTORIES.get(session_id)
    if history is None:
        return
    if len(history.messages) > _MAX_HISTORY_MESSAGES:
        history.messages = history.messages[-_MAX_HISTORY_MESSAGES:]


def _format_context(docs_with_scores: List[Tuple[Document, float]]) -> str:
    parts: List[str] = []
    for doc, _ in docs_with_scores:
        parts.append(doc.page_content)
    return "\n\n---\n\n".join(parts)


def _extract_sources(docs_with_scores: List[Tuple[Document, float]]) -> List[Dict[str, Any]]:
    """
    Aggregate sources so that each document appears once,
    with a deduplicated, sorted list of pages.
    """
    by_document: Dict[str, set] = {}
    for doc, _ in docs_with_scores:
        source = doc.metadata.get("source", "unknown")
        raw_page = doc.metadata.get("page")
        try:
            page = int(raw_page) if raw_page is not None else 0
        except (TypeError, ValueError):
            page = 0
        if source not in by_document:
            by_document[source] = set()
        by_document[source].add(page)

    sources: List[Dict[str, Any]] = []
    for source, pages in by_document.items():
        sorted_pages = sorted(p for p in pages if p is not None)
        sources.append({"document": source, "pages": sorted_pages})
    return sources


def answer_question_with_rag(
    question: str,
    session_id: str,
    k: int = 5,
) -> Dict[str, Any]:
    """
    High-level RAG pipeline:
    - retrieve documents
    - format context
    - run LLM
    - return answer + sources
    """
    docs_with_scores = retrieve_relevant_documents(question, k=k)
    print(f"Retrieved {len(docs_with_scores)} documents for question: '{question}'")
    context = _format_context(docs_with_scores)

    runnable = build_qa_runnable()
    runnable_with_history = RunnableWithMessageHistory(
        runnable,
        _get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
    answer = runnable_with_history.invoke(
        {"context": context, "question": question},
        config={"configurable": {"session_id": session_id}},
    )
    _trim_session_history(session_id)

    clean_answer = answer.strip()
    
    if (
        FALLBACK_MESSAGE in clean_answer or 
        NON_HEALTHCARE_MESSAGE in clean_answer or 
        GREETING_MESSAGE in clean_answer
    ):
        return {"answer": answer, "sources": []}

    sources = _extract_sources(docs_with_scores)
    return {"answer": answer, "sources": sources}
