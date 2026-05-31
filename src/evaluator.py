"""RAG evaluation utilities: precision@k, chunk stats, and faithfulness/relevancy scoring."""

from typing import Any, Callable, Dict, List


def compute_precision_at_k(
    retrieved_ids: List[str], relevant_ids: List[str], k: int = 3
) -> float:
    """Compute Precision@k for a single query.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: List of relevant document IDs.
        k: Number of top retrieved items to consider.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_set)
    return hits / k


def compute_chunk_stats(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistics over a list of chunks.

    Args:
        chunks: List of chunk dicts with at least a "char_count" key.

    Returns:
        Dict with "chunk_count" and "avg_chunk_size".
    """
    if not chunks:
        return {"chunk_count": 0, "avg_chunk_size": 0.0}
    total_chars = sum(chunk["char_count"] for chunk in chunks)
    count = len(chunks)
    return {"chunk_count": count, "avg_chunk_size": total_chars / count}


def _score_faithfulness(
    answer: str,
    context_chunks: List[str],
    llm_client: Any,
) -> float:
    """Score faithfulness: does the answer stick to the provided context?

    Uses an LLM prompt to judge if each claim in the answer is supported by
    the retrieved context. Returns a float 0–1.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    context_text = "\n\n".join(
        f"[{i}] {chunk}" for i, chunk in enumerate(context_chunks)
    )
    system_msg = SystemMessage(
        content=(
            "You are an evaluation judge. Your task is to rate how faithful "
            "an answer is to the provided context. A faithful answer only "
            "makes claims that are directly supported by the context. "
            "Return ONLY a number between 0.0 and 1.0:\n"
            "- 1.0: All claims are directly supported by the context.\n"
            "- 0.5: Some claims are supported, some are not.\n"
            "- 0.0: The answer contradicts or is unrelated to the context.\n"
            "Return the number on its own line, nothing else."
        ),
    )
    user_msg = HumanMessage(
        content=(
            f"Context:\n{context_text}\n\n"
            f"Answer to judge:\n{answer}\n\n"
            f"Faithfulness score (0.0-1.0):"
        ),
    )
    response = llm_client.invoke([system_msg, user_msg])
    try:
        score = float(response.content.strip().split("\n")[0])
        return max(0.0, min(1.0, score))
    except (ValueError, AttributeError):
        return 0.0


def _score_answer_relevancy(
    query: str,
    answer: str,
    llm_client: Any,
) -> float:
    """Score answer relevancy: how well the answer addresses the query.

    Uses an LLM prompt to judge if the answer is on-topic and complete.
    Returns a float 0–1.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    system_msg = SystemMessage(
        content=(
            "You are an evaluation judge. Your task is to rate how relevant "
            "an answer is to the given query. A relevant answer directly "
            "addresses all parts of the question with appropriate detail.\n"
            "Return ONLY a number between 0.0 and 1.0:\n"
            "- 1.0: The answer fully addresses the query.\n"
            "- 0.5: The answer partially addresses the query.\n"
            "- 0.0: The answer is off-topic or irrelevant.\n"
            "Return the number on its own line, nothing else."
        ),
    )
    user_msg = HumanMessage(
        content=(
            f"Query: {query}\n\n"
            f"Answer to judge:\n{answer}\n\n"
            f"Answer relevancy score (0.0-1.0):"
        ),
    )
    response = llm_client.invoke([system_msg, user_msg])
    try:
        score = float(response.content.strip().split("\n")[0])
        return max(0.0, min(1.0, score))
    except (ValueError, AttributeError):
        return 0.0


def evaluate_strategy(
    strategy_name: str,
    queries: List[Dict[str, Any]],
    collection: Any,
    embeddings: Any,
    llm_client: Any,
    retriever_fn: Callable[[Any, str, int, Any], List[Dict[str, Any]]],
    generator_fn: Callable[[str, List[Dict[str, Any]], Any], str],
) -> Dict[str, Any]:
    """Evaluate a single RAG strategy across all queries.

    For each query:
      1. Retrieve top-k chunks.
      2. Generate an answer.
      3. Compute Precision@3.
      4. Score faithfulness (LLM judge).
      5. Score answer relevancy (LLM judge).

    Args:
        strategy_name: Name of the chunking strategy.
        queries: List of query dicts with keys: query, ground_truth, relevant_doc_ids.
        collection: ChromaDB collection object.
        embeddings: Embedding model.
        llm_client: LLM client for generation and scoring.
        retriever_fn: (collection, query_text, top_k, embeddings) -> results.
        generator_fn: (query_text, retrieved_chunks, llm_client) -> answer string.

    Returns:
        Dict with strategy, precision@3, faithfulness, answer_relevancy,
        chunk_count, avg_chunk_size.
    """
    precisions: List[float] = []
    faithfulness_scores: List[float] = []
    relevancy_scores: List[float] = []

    for q in queries:
        query_text = q["query"]
        relevant_doc_ids = q.get("relevant_doc_ids", [])

        retrieved = retriever_fn(collection, query_text, top_k=3, embeddings=embeddings)
        retrieved_ids = [r["id"] for r in retrieved]
        retrieved_texts = [r["text"] for r in retrieved]

        # Precision@3
        precision = compute_precision_at_k(retrieved_ids, relevant_doc_ids, k=3)
        precisions.append(precision)

        # Generate answer
        answer = generator_fn(query_text, retrieved, llm_client)

        # Faithfulness and Answer Relevancy via LLM judge
        if retrieved_texts:
            faith = _score_faithfulness(answer, retrieved_texts, llm_client)
        else:
            faith = 0.0
        faithfulness_scores.append(faith)

        rel = _score_answer_relevancy(query_text, answer, llm_client)
        relevancy_scores.append(rel)

    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    avg_faithfulness = (
        sum(faithfulness_scores) / len(faithfulness_scores)
        if faithfulness_scores
        else 0.0
    )
    avg_relevancy = (
        sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0.0
    )

    return {
        "strategy": strategy_name,
        "precision@3": round(avg_precision, 4),
        "faithfulness": round(avg_faithfulness, 4),
        "answer_relevancy": round(avg_relevancy, 4),
        "chunk_count": 0,
        "avg_chunk_size": 0.0,
    }
