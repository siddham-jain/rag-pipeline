"""Visualization script for RAG chunking strategy comparison.

Produces:
- results/chunk_size_distribution.png : overlaid histogram of chunk sizes
- results/comparison_table.md      : markdown table from comparison.csv
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

from config import (
    BREAKPOINT_THRESHOLD_TYPE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    ESSAYS_DIR,
    MIN_CHUNK_LENGTH,
    RESULTS_DIR,
)
from src.chunkers.fixed_size import chunk_fixed_size
from src.chunkers.recursive import chunk_recursive
from src.chunkers.semantic import chunk_semantic
from src.data_loader import load_essays


def ensure_results_dir() -> None:
    """Create the results directory if it does not already exist."""
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)


def collect_chunk_sizes(text: str) -> dict[str, list[int]]:
    """Run all three chunkers on *text* and return character-count lists."""
    fixed = [c["char_count"] for c in chunk_fixed_size(text, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LENGTH)]
    recursive = [c["char_count"] for c in chunk_recursive(text, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LENGTH)]
    semantic = [c["char_count"] for c in chunk_semantic(text, BREAKPOINT_THRESHOLD_TYPE, MIN_CHUNK_LENGTH)]
    return {"fixed_size": fixed, "recursive": recursive, "semantic": semantic}


def plot_chunk_size_distribution(sizes: dict[str, list[int]]) -> None:
    """Generate and save the overlaid chunk-size histogram."""
    plt.figure(figsize=(10, 6))
    sns.histplot(sizes["fixed_size"], color="blue", alpha=0.5, kde=True, bins=40, label="Fixed-size")
    sns.histplot(sizes["recursive"], color="green", alpha=0.5, kde=True, bins=40, label="Recursive")
    sns.histplot(sizes["semantic"], color="orange", alpha=0.5, kde=True, bins=40, label="Semantic")
    plt.xlabel("Chunk Size (characters)")
    plt.ylabel("Density")
    plt.title("Chunk Size Distribution by Strategy")
    plt.legend()
    plt.tight_layout()

    out_path = Path(RESULTS_DIR) / "chunk_size_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved histogram to {out_path}")


def build_comparison_table() -> str:
    """Read comparison.csv and return a formatted markdown table string."""
    csv_path = Path(RESULTS_DIR) / "comparison.csv"
    df = pd.read_csv(csv_path)

    # Rename / select columns to match requested output
    column_map = {
        "Strategy": "Strategy",
        "Precision@3": "Precision@3",
        "Faithfulness": "Faithfulness",
        "Answer Relevancy": "Answer Relevancy",
        "Avg Chunk Size": "Avg Chunk Size",
        "Chunk Count": "Chunk Count",
    }
    # Keep only columns that actually exist in the CSV
    available = {k: v for k, v in column_map.items() if v in df.columns}
    df = df[list(available.values())].copy()
    df.rename(columns={v: k for k, v in available.items()}, inplace=True)

    # Round floats to 3 decimal places
    for col in df.columns:
        if df[col].dtype.kind in "fc":  # float or complex
            df[col] = df[col].round(3)

    return df.to_markdown(index=False)


def save_comparison_table(md: str) -> None:
    """Write the markdown table to file and print it to stdout."""
    out_path = Path(RESULTS_DIR) / "comparison_table.md"
    out_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nSaved comparison table to {out_path}")


def main() -> None:
    """Orchestrate visualization generation."""
    load_dotenv()          # make OPENAI_API_KEY available for semantic chunker
    ensure_results_dir()

    essays = load_essays(ESSAYS_DIR)
    all_text = "\n\n".join(essays)

    sizes = collect_chunk_sizes(all_text)
    plot_chunk_size_distribution(sizes)

    md_table = build_comparison_table()
    save_comparison_table(md_table)


if __name__ == "__main__":
    main()
