"""Semantic text chunking strategy using embedding similarity."""

from typing import List, Dict

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential

from config import RETRY_MAX_ATTEMPTS, RETRY_MIN_WAIT, RETRY_MAX_WAIT


@retry(
    stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
    reraise=True,
)
def chunk_semantic(
    text: str, breakpoint_threshold_type: str, min_length: int
) -> List[Dict[str, any]]:
    """Split text into semantically coherent chunks using embedding similarity.

    Uses ``langchain_experimental.text_splitter.SemanticChunker`` with
    ``OpenAIEmbeddings`` to compute sentence embeddings and detect semantic
    breakpoints based on cosine-similarity drops. The breakpoint threshold
    type (e.g. ``"percentile"``) determines how aggressively chunks are split.

    Because every sentence is embedded during chunking, this strategy incurs
    higher API cost than fixed-size or recursive approaches. It may also
    produce fewer, larger chunks when the text is topically uniform.

    Args:
        text: Input text to chunk.
        breakpoint_threshold_type: Threshold strategy for semantic breakpoints
            (e.g. ``"percentile"``, ``"standard_deviation"``, ``"interquartile"``).
        min_length: Minimum character length; chunks shorter than this are
            filtered out.

    Returns:
        List of chunk dictionaries, each containing:
            - id: unique chunk identifier (e.g. "semantic_chunk_0")
            - text: the chunk text content
            - strategy: always "semantic"
            - chunk_index: zero-based index of the chunk
            - char_count: number of characters in the chunk

    Raises:
        RuntimeError: If chunking fails after retries, with a meaningful message.
    """
    if not text:
        return []

    try:
        embeddings = OpenAIEmbeddings()
        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
        )
        raw_chunks = splitter.split_text(text)
    except Exception as exc:
        raise RuntimeError(
            f"Semantic chunking failed: {type(exc).__name__}: {exc}"
        ) from exc

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        if len(chunk_text) >= min_length:
            chunks.append({
                "id": f"semantic_chunk_{i}",
                "text": chunk_text,
                "strategy": "semantic",
                "chunk_index": i,
                "char_count": len(chunk_text),
            })

    return chunks
