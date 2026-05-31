"""Fixed-size text chunking strategy."""

from typing import List, Dict

from langchain_text_splitters import CharacterTextSplitter


def chunk_fixed_size(
    text: str, chunk_size: int, chunk_overlap: int, min_length: int
) -> List[Dict[str, any]]:
    """Split text into fixed-size chunks with optional overlap.

    Uses langchain_text_splitters.CharacterTextSplitter with an empty separator,
    meaning chunks are purely size-based without regard to word or sentence boundaries.

    Args:
        text: Input text to chunk.
        chunk_size: Maximum character count per chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.
        min_length: Minimum character length; chunks shorter than this are filtered out.

    Returns:
        List of chunk dictionaries, each containing:
            - id: unique chunk identifier (e.g. "fixed_size_chunk_0")
            - text: the chunk text content
            - strategy: always "fixed_size"
            - chunk_index: zero-based index of the chunk
            - char_count: number of characters in the chunk
    """
    if not text:
        return []

    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    raw_chunks = splitter.split_text(text)

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        if len(chunk_text) >= min_length:
            chunks.append({
                "id": f"fixed_size_chunk_{i}",
                "text": chunk_text,
                "strategy": "fixed_size",
                "chunk_index": i,
                "char_count": len(chunk_text),
            })

    return chunks