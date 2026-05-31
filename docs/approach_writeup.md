# Approach & validation

## What I was trying to learn

I wanted to see whether chunking strategy actually matters for RAG quality, or if any reasonable splitter is good enough once you have decent embeddings. The rest of the pipeline stays fixed so the comparison is fair: same essays, same embedding model, same vector store, same retriever, same generator, same test questions.

## Approach

I built three parallel indexes in ChromaDB—one per chunking strategy—and ran every query through each one end to end: chunk → embed → retrieve top 3 → generate an answer.

The three strategies were deliberate extremes plus a middle ground:

- **Fixed-size** splits on character count with overlap and does not respect sentence boundaries. It is the naive baseline—cheap and predictable, but it can cut ideas in half.
- **Recursive** uses LangChain’s recursive splitter with the same target size and overlap, but it tries to break on paragraphs and sentences first. Same rough chunk budget as fixed-size, hopefully cleaner boundaries.
- **Semantic** uses embedding similarity between sentences to find topic shifts and split there. It costs more at index time because chunking itself calls the embeddings API, and it tends to produce fewer, larger chunks.

I used Paul Graham essays as the corpus because they are long-form, argumentative prose—exactly the kind of text where bad splits hurt retrieval. Each chunk ID is prefixed with the essay name so I could label which chunks should come back for each question.

## Models and settings

- **Embeddings:** `text-embedding-3-small` for indexing and query vectors.
- **Generation:** `gpt-4o-mini` with temperature 0 so runs are repeatable during evaluation.
- **Chunking defaults:** 1000 characters with 200 overlap for fixed and recursive; semantic uses a percentile breakpoint on embedding distances. Anything under 50 characters gets dropped.
- **Retrieval:** top 3 chunks only—enough context for short answers without drowning the model.

I kept generation prompts strict: answer only from the provided context, and say when there is not enough information. That way failures are easier to attribute to retrieval or chunking rather than the model freelancing.

## Decisions and trade-offs

**Why control everything except chunking.** If I changed models or k per strategy, I would not know what caused a score to move. The trade-off is that this does not tell you the best possible RAG stack—only which chunker fits this particular setup.

**Why hand-written queries.** Five questions cover factual lookup, multi-hop reasoning, negation, cross-essay comparison, and summarisation. That is a small set, but it is enough to stress different failure modes. The downside is low statistical power; one bad retrieval on a hard question moves the average a lot.

**Why Precision@3 with manual chunk IDs.** I labeled `relevant_doc_ids` per query after reading the essays and knowing which chunks actually contain the answer. That gives a concrete retrieval metric that does not depend on another model’s opinion. The catch is labeling effort and the fact that “relevant” is somewhat subjective for broad summary questions.

**Why LLM-as-judge for faithfulness and relevancy.** I score whether the answer sticks to the retrieved chunks and whether it addresses the question, using the same `gpt-4o-mini` with simple rubric prompts. It is cheaper and faster than running a full human eval loop on every answer, but judges can be lenient or inconsistent, so I treat those scores as directional, not ground truth. (Ragas is in the dependencies and the main script comments refer to it, but the evaluation path I actually run is these custom judges in `evaluator.py`.)

**Semantic vs recursive in practice.** Semantic indexing is slower and more expensive up front, but it often wins on retrieval precision with fewer chunks to store. Recursive is a solid default when you want structure-aware splits without extra API calls during ingest.

## How I validated

1. **Retrieval:** For each query and strategy, I checked whether the top-3 retrieved chunk IDs overlapped my labeled relevant set, then averaged Precision@3 across all five queries.

2. **Generation quality:** For the same retrieved context, I generated answers and scored faithfulness (claims supported by context) and answer relevancy (does it actually answer the question).

3. **Chunking health:** I logged chunk count and average chunk size per strategy and used `visualize.py` to compare size distributions—mostly to sanity-check that semantic was really producing larger, fewer chunks.

4. **Cross-strategy comparison:** I wrote everything to `results/comparison.csv` and looked for patterns, not single lucky runs. In my run, semantic came out ahead on Precision@3; fixed-size was clearly weakest, especially on the factual question where mid-sentence splits seemed to hurt both retrieval and faithfulness.

5. **Failure analysis:** I picked the worst query–strategy pair and wrote up why it failed (`results/deployment_recommendation.md`)—that grounded the recommendation in a concrete example instead of only staring at averages.

## What I would do differently

With more time I would add hybrid retrieval for negation-style questions, a reranker on top of vector search, and either human spot-checks or a second judge model on a sample of answers. The current setup is enough to choose a chunking strategy for this corpus, but I would not treat the faithfulness numbers as production SLAs without more validation.
