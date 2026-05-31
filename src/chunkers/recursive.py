"""Recursive text chunker for the RAG pipeline."""
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_recursive(text: str, chunk_size: int, chunk_overlap: int, min_length: int) -> list[dict]:
    """Split text into chunks using recursive character-based splitting.

    Respects natural text boundaries by attempting splits in order of preference:
    paragraph boundaries ("\\n\\n"), then sentence boundaries ("\\n"), then
    mid-sentence (". "), and finally individual words (" "). This preserves
    semantic coherence better than fixed-size chunking.

    Args:
        text: The input text to chunk.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
        min_length: Minimum character length; chunks shorter than this are filtered out.

    Returns:
        List of chunk dictionaries, each containing:
        - id: Unique identifier (format: "recursive_chunk_{index}")
        - text: The chunk content
        - strategy: Always "recursive"
        - chunk_index: Zero-based index of the chunk
        - char_count: Number of characters in the chunk
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    raw_chunks = splitter.split_text(text)

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        if len(chunk_text) >= min_length:
            chunks.append({
                "id": f"recursive_chunk_{i}",
                "text": chunk_text,
                "strategy": "recursive",
                "chunk_index": i,
                "char_count": len(chunk_text),
            })

    return chunks