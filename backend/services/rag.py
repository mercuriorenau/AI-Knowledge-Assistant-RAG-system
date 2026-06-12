import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.config import Settings
from backend.models.schemas import QueryResponse
from backend.services.metrics import metrics
from backend.services.retrieval import pairs_to_sources, retrieve_chunks

SYSTEM_PROMPT = """You are a careful assistant.
You must answer ONLY using the provided CONTEXT.
If the answer is not in the context, reply exactly: "I don't know."
Do not invent facts, names, dates, or details not supported by the context.
Keep answers concise and grounded."""


def answer_question(
    question: str,
    settings: Settings,
    k: int | None = None,
) -> QueryResponse:
    started = time.perf_counter()
    try:
        pairs, context = retrieve_chunks(question, settings, k=k)
        sources = pairs_to_sources(pairs)

        if not context.strip():
            return QueryResponse(
                answer=(
                    "I don't know. No relevant information found in the indexed documents. "
                    "Upload documents or lower MIN_RELEVANCE_SCORE if needed."
                ),
                sources=[],
                context_empty=True,
            )

        user_content = (
            f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\n"
            "Answer using the context only."
        )

        llm = ChatOpenAI(
            model=settings.chat_model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key or None,
        )
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ]
        )
        answer = response.content if isinstance(response.content, str) else str(response.content)
        return QueryResponse(answer=answer.strip(), sources=sources, context_empty=False)
    finally:
        metrics.record_query((time.perf_counter() - started) * 1000.0)
