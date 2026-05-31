"""Main entry point for the RAG pipeline comparison."""

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from config import *
from queries import QUERIES
from src import data_loader, embedder, evaluator, generator, retriever, store
from src.chunkers.fixed_size import chunk_fixed_size
from src.chunkers.recursive import chunk_recursive
from src.chunkers.semantic import chunk_semantic

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def _retriever_fn(collection, query_text, top_k, embeddings):
    """Wrapper matching evaluate_strategy's retriever signature."""
    return retriever.retrieve(query_text, collection, embeddings, top_k)


def _generator_fn(query_text, retrieved_chunks, llm_client):
    """Wrapper matching evaluate_strategy's generator signature."""
    chunk_texts = [r["text"] for r in retrieved_chunks]
    return generator.generate_answer(query_text, chunk_texts, LLM_MODEL, TEMPERATURE)


def _rewrite_chunk_ids(chunks, doc_prefix):
    """Rewrite chunk IDs to include the essay name for ground-truth matching."""
    for chunk in chunks:
        chunk["id"] = f"{doc_prefix}_chunk_{chunk['chunk_index']}"
    return chunks


def main():
    """Orchestrate the full RAG pipeline end-to-end."""
    load_dotenv()

    # Create results directory if missing
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load essays
    logging.info("Loading essays from %s", ESSAYS_DIR)
    essays = data_loader.load_essays(ESSAYS_DIR)
    essay_paths = sorted(Path(ESSAYS_DIR).glob("*.txt"))

    # 2. Initialize embeddings
    logging.info("Initializing embeddings (%s)", EMBEDDING_MODEL)
    embeddings = embedder.get_embeddings()

    # 3. Initialize ChromaDB client
    logging.info("Initializing ChromaDB client at %s", CHROMA_PERSIST_DIR)
    chroma_client = store.get_chroma_client(CHROMA_PERSIST_DIR)

    strategies = ["fixed_size", "recursive", "semantic"]

    # Step 5: For each strategy: chunk, create/recreate collection, add docs, compute stats
    strategy_chunks = {}
    strategy_stats = {}
    strategy_collections = {}

    for strategy_name in strategies:
        logging.info("Chunking essays with strategy: %s", strategy_name)

        chunks = []
        for essay_text, essay_path in zip(essays, essay_paths):
            doc_prefix = essay_path.stem
            if strategy_name == "fixed_size":
                strategy_chunks_list = chunk_fixed_size(
                    essay_text, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LENGTH
                )
            elif strategy_name == "recursive":
                strategy_chunks_list = chunk_recursive(
                    essay_text, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LENGTH
                )
            else:  # semantic
                strategy_chunks_list = chunk_semantic(
                    essay_text, BREAKPOINT_THRESHOLD_TYPE, MIN_CHUNK_LENGTH
                )
            strategy_chunks_list = _rewrite_chunk_ids(strategy_chunks_list, doc_prefix)
            chunks.extend(strategy_chunks_list)

        collection_name = COLLECTIONS[strategy_name]
        logging.info("Creating collection: %s", collection_name)
        collection = store.create_collection(chroma_client, collection_name, embeddings)

        logging.info("Adding %d chunks to collection: %s", len(chunks), collection_name)
        store.add_documents(collection, chunks, embeddings)

        stats = evaluator.compute_chunk_stats(chunks)
        strategy_chunks[strategy_name] = chunks
        strategy_stats[strategy_name] = stats
        strategy_collections[strategy_name] = collection

        logging.info(
            "Strategy %s: %d chunks, avg size %.1f chars",
            strategy_name,
            stats["chunk_count"],
            stats["avg_chunk_size"],
        )

    # Step 6: For each query, for each strategy: retrieve, generate, compute precision@3
    query_strategy_precisions = {s: [] for s in strategies}

    for query_item in QUERIES:
        query_text = query_item["query"]
        relevant_ids = query_item.get("relevant_doc_ids", [])

        for strategy_name in strategies:
            collection = strategy_collections[strategy_name]
            retrieved = retriever.retrieve(query_text, collection, embeddings, TOP_K)
            retrieved_ids = [r["id"] for r in retrieved]
            chunk_texts = [r["text"] for r in retrieved]

            # Generate answer
            answer = generator.generate_answer(query_text, chunk_texts, LLM_MODEL, TEMPERATURE)

            # Compute precision@3
            precision = evaluator.compute_precision_at_k(retrieved_ids, relevant_ids, k=3)
            query_strategy_precisions[strategy_name].append(precision)

    # Step 7: Evaluate each strategy with ragas
    llm_client = ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    strategy_eval_results = {}

    for strategy_name in strategies:
        logging.info("Evaluating strategy with ragas: %s", strategy_name)
        collection = strategy_collections[strategy_name]
        result = evaluator.evaluate_strategy(
            strategy_name=strategy_name,
            queries=QUERIES,
            collection=collection,
            embeddings=embeddings,
            llm_client=llm_client,
            retriever_fn=_retriever_fn,
            generator_fn=_generator_fn,
        )
        strategy_eval_results[strategy_name] = result

    # Step 8: Compile results
    results = []
    for strategy_name in strategies:
        avg_precision = sum(query_strategy_precisions[strategy_name]) / len(
            query_strategy_precisions[strategy_name]
        )
        eval_result = strategy_eval_results[strategy_name]
        stats = strategy_stats[strategy_name]

        results.append(
            {
                "strategy": strategy_name,
                "precision@3": avg_precision,
                "faithfulness": eval_result["faithfulness"],
                "answer_relevancy": eval_result["answer_relevancy"],
                "chunk_count": stats["chunk_count"],
                "avg_chunk_size": stats["avg_chunk_size"],
            }
        )

    # Step 9: Save to results/comparison.csv
    df = pd.DataFrame(results)
    csv_path = Path(RESULTS_DIR) / "comparison.csv"
    df.to_csv(csv_path, index=False)
    logging.info("Results saved to %s", csv_path)

    # Step 10: Print summary table
    print("\n" + "=" * 80)
    print("RAG PIPELINE COMPARISON RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
