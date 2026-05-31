# Approach write-up

## Corpus and pipeline

I used five Paul Graham essays (`data/essays/`) as the long-form corpus—each well over 2,000 words combined. The pipeline loads them, chunks with one of three strategies, embeds with `text-embedding-3-small`, stores in ChromaDB, retrieves top 3 chunks per query, and answers with `gpt-4o-mini` (temperature 0) using a strict “context only” prompt.

## Chunking strategies

- **Fixed-size** — character windows (1000 chars, 200 overlap); baseline, can split mid-sentence.
- **Recursive** — same target size, breaks on paragraphs/sentences first.
- **Semantic** — LangChain `SemanticChunker` with percentile breakpoints on embedding similarity; fewer, larger, topic-coherent chunks, extra embedding cost at ingest.

Each strategy gets its own Chroma collection so comparisons stay apples-to-apples. Chunk IDs include the essay name for Precision@3 labeling.

## Queries and metrics

Five hand-written queries in `queries.py` cover factual, multi-hop, negation, comparison, and summarisation. Each has `relevant_doc_ids` for retrieval scoring.

Per strategy, `main.py` reports:

- **Precision@3** — labeled relevant chunks in the top 3 retrieved
- **Faithfulness** — LLM judge: answer supported by retrieved context
- **Answer relevancy** — LLM judge: answer addresses the query
- **Chunk count** and **average chunk size**

Outputs: `results/comparison.csv`, `results/comparison_table.md`, `results/chunk_size_distribution.png` (via `visualize.py`).

## Trade-offs

I held embeddings, retriever, generator, and queries fixed so differences trace to chunking only. The eval set is small (five queries), so scores are indicative, not statistically tight. LLM judges are fast to run but noisier than human grading. Manual `relevant_doc_ids` are objective for retrieval but take judgment on broad summary questions.

## Results (reference run)

Semantic led on Precision@3 (0.47) with the fewest chunks (92) and largest average size (~1,650 chars), matching recursive on faithfulness/relevancy (0.6 / 0.8). Fixed-size was weakest overall (Precision@3 0.2, faithfulness 0.3).

**Failure case:** factual query (“key to doing work you truly love”) + fixed-size — fragmented chunks, Precision@3 0.2, faithfulness 0.3. Details in `results/deployment_recommendation.md`.

## Deployment recommendation

Deploy **semantic chunking** for this essay-style corpus: best retrieval precision and fewer chunks to embed/store. Use **recursive** if you want structure-aware splits without embedding cost during ingest.

Hallucination risk remains (faithfulness ~0.6). Mitigate with a reranker on retrieved chunks, cite-or-abstain prompting, hybrid/keyword search for negation queries, and low-confidence routing when retrieval scores are weak. Monitor Precision@3 and faithfulness on new content.
