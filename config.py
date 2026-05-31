"""Centralized configuration constants for the RAG pipeline."""

# Chunking parameters
CHUNK_SIZE = 1000          # characters for fixed-size chunker
CHUNK_OVERLAP = 200        # character overlap for fixed-size and recursive
BREAKPOINT_THRESHOLD_TYPE = "percentile"  # for semantic chunker
MIN_CHUNK_LENGTH = 50      # filter out too-short chunks

# Embedding and retrieval
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 3                  # number of chunks to retrieve

# Generation
LLM_MODEL = "gpt-4o-mini"
MAX_TOKENS = 1024
TEMPERATURE = 0.0          # deterministic generation for evaluation

# Storage paths
CHROMA_PERSIST_DIR = "./chroma_db"
RESULTS_DIR = "./results"
ESSAYS_DIR = "./data/essays"

# ChromaDB collection names
COLLECTIONS = {
    "fixed_size": "rag_fixed_size",
    "recursive": "rag_recursive",
    "semantic": "rag_semantic",
}

# API retry config
RETRY_MAX_ATTEMPTS = 3
RETRY_MIN_WAIT = 2         # seconds
RETRY_MAX_WAIT = 10        # seconds