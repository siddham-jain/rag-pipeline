"""Retriever module: embed query and search ChromaDB collection directly."""
from typing import Any


def retrieve(
    query: str,
    collection: Any,
    embeddings: Any,
    top_k: int,
) -> list[dict]:
    """Embed a query and retrieve the top-k most similar documents from a ChromaDB collection.

    Args:
        query: The user query string.
        collection: A ChromaDB Collection object.
        embeddings: An embedding model with an ``embed_query`` method (e.g. OpenAIEmbeddings).
        top_k: Number of results to return.

    Returns:
        A list of dicts with keys ``id``, ``text``, ``metadata``, and ``distance``.
        Returns an empty list if the collection contains no documents.
    """
    if collection.count() == 0:
        return []

    query_embedding = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i in range(len(ids)):
        retrieved.append(
            {
                "id": ids[i],
                "text": documents[i],
                "metadata": metadatas[i] if metadatas else {},
                "distance": distances[i] if distances else None,
            }
        )

    return retrieved
