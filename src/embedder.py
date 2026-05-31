"""Embedding provider with retry resilience."""
from langchain_openai import OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential

import config


@retry(
    stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=config.RETRY_MIN_WAIT, max=config.RETRY_MAX_WAIT),
)
def get_embeddings() -> OpenAIEmbeddings:
    """Return a configured OpenAIEmbeddings instance with API retry logic."""
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
