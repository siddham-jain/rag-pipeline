"""Generator module: build a prompt and call an LLM via LangChain ChatOpenAI."""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential

import config


SYSTEM_PROMPT = (
    "Answer the following question based only on the provided context. "
    "If the context doesn't contain enough information to answer the question, say so explicitly. "
    "Do not make up information."
)


@retry(
    stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=config.RETRY_MIN_WAIT, max=config.RETRY_MAX_WAIT),
)
def _call_llm(messages: list, model: str, temperature: float) -> str:
    """Invoke ChatOpenAI with retry resilience."""
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=config.MAX_TOKENS,
    )
    response = llm.invoke(messages)
    return response.content


def generate_answer(
    query: str,
    context_chunks: list[str],
    llm_model: str,
    temperature: float = 0.0,
) -> str:
    """Generate an answer from an LLM using labeled context chunks.

    Args:
        query: The user question.
        context_chunks: List of text chunks to use as context.
        llm_model: Name of the OpenAI chat model to use.
        temperature: Sampling temperature (default 0.0 for deterministic output).

    Returns:
        The generated answer string. If ``context_chunks`` is empty, returns a
        fixed insufficient-context message without calling the API.
    """
    if not context_chunks:
        return "Insufficient context to answer this question."

    labeled_context = "\n\n".join(
        f"[{i + 1}] {chunk}" for i, chunk in enumerate(context_chunks)
    )

    user_message = (
        f"Context:\n{labeled_context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    return _call_llm(messages, model=llm_model, temperature=temperature)
