"""ChromaDB store utilities (direct chromadb, no LangChain wrapper)."""
from typing import Any, Dict, List

import chromadb
from chromadb.api.models.Collection import Collection


def get_chroma_client(persist_dir: str) -> chromadb.PersistentClient:
    """Return a ChromaDB persistent client for the given directory."""
    return chromadb.PersistentClient(path=persist_dir)


def create_collection(client: chromadb.PersistentClient, collection_name: str, embeddings: Any) -> Collection:
    """Create or recreate a ChromaDB collection.

    Deletes the collection if it already exists to avoid duplicates on re-run.
    Does not pass an embedding_function to ChromaDB since embeddings are handled manually.
    """
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass  # collection didn't exist
    return client.create_collection(name=collection_name)


def add_documents(collection: Collection, chunks: List[Dict[str, Any]], embeddings: Any) -> None:
    """Embed chunk texts and add them to the collection.

    Args:
        collection: ChromaDB collection.
        chunks: List of chunk dicts with keys: id, text, strategy, chunk_index, char_count.
        embeddings: Embedding model with embed_documents() method.
    """
    texts = [chunk["text"] for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]
    metadatas = [
        {
            "strategy": chunk["strategy"],
            "chunk_index": chunk["chunk_index"],
            "char_count": chunk["char_count"],
        }
        for chunk in chunks
    ]
    vectors = embeddings.embed_documents(texts)
    collection.add(ids=ids, embeddings=vectors, metadatas=metadatas, documents=texts)


def query_collection(collection: Collection, query_text: str, top_k: int, embeddings: Any) -> List[Dict[str, Any]]:
    """Embed a query, search the collection, and return formatted results.

    Returns:
        List of dicts with keys: id, text, metadata, distance.
    """
    query_vector = embeddings.embed_query(query_text)
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)
    output = []
    for i in range(len(results["ids"][0])):
        output.append(
            {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )
    return output
