# RAG Pipeline Deployment Recommendation

## Failure Case Analysis

The worst-performing query-strategy pair was the **factual query** with the **fixed_size** strategy. The generated answer scored a Faithfulness of 0.3 and the fixed-size strategy achieved the lowest overall Precision@3 of 0.200.

**Query:** "According to Paul Graham, what is the key to doing work you truly love?"

**Strategy:** fixed_size

**Why it failed:** The fixed-size chunker splits text at arbitrary character boundaries without regard for semantic coherence. This caused the retrieved chunks to contain partial sentences and fragmented ideas, making it difficult for the LLM to reconstruct the author's intended meaning. The answer generated from fixed-size chunks was less faithful to the source material because the context window was filled with sentence fragments rather than complete thoughts. In contrast, the recursive chunker (Faithfulness 0.6) and semantic chunker (Faithfulness 0.6) both respected paragraph and sentence boundaries, producing more coherent context for the LLM to work from.

This failure illustrates a fundamental weakness of naive chunking — when text is split mid-sentence, the embedding vectors become noisier and retrieval quality degrades, which cascades into lower generation quality. The Precision@3 of 0.200 for fixed-size confirms that the top-3 retrieved chunks rarely contained the correct answer passages.

## Deployment Recommendation

1. **The semantic chunking strategy is recommended for deployment**, achieving the best Precision@3 (0.467 vs 0.333 and 0.200) on multi-topic essay-style content, while maintaining strong faithfulness (0.6) and answer relevancy (0.8) tied with the recursive strategy. Semantic chunking produced 92 chunks averaging 1649.8 characters, far fewer than fixed-size (192 chunks) and recursive (198 chunks), which means lower embedding and storage costs.

2. Semantic chunking outperformed because it uses embedding similarity to find natural topic boundaries, creating larger but more coherent chunks. This means each chunk contains a complete idea, making both retrieval and generation more effective. The recursive chunker performed respectably (0.333 Precision@3, 0.6 Faithfulness) and is a good fallback that requires no extra API calls during chunking.

3. **Hallucination risk remains** even with the best strategy. At Faithfulness 0.6 and Answer Relevancy 0.8, approximately 40% of generated content may still introduce unsupported claims, particularly for multi-hop and negation queries that require connecting information across distant chunks or reasoning about what is NOT stated.

4. **To mitigate hallucination risk in production**, implement a cross-encoder reranker on top of the semantic retriever to improve precision from 0.467 toward 0.7+, use strict prompt instructions that require the model to cite specific chunk references for each claim, and add a guard that flags answers with retrieval similarity below a threshold (e.g., cosine distance > 0.3) for human review.

5. **Monitor Faithfulness and Precision@3 weekly** in production. Set up automated alerts when Faithfulness drops below 0.5 or Precision@3 falls below 0.3. Periodically review the worst-performing query types (negation queries are particularly challenging for vector-only retrieval) and consider adding keyword/hybrid search for those query categories. Track chunk size distribution drift to ensure new content doesn't degrade the semantic chunker's performance.