"""Load Paul Graham essays from a directory."""
import os
from pathlib import Path
from typing import List


def load_essays(directory: str) -> List[str]:
    """Read all .txt files from the given directory and return their contents.

    Args:
        directory: Path to directory containing essay .txt files.

    Returns:
        List of essay text strings, one per file.
    """
    essay_dir = Path(directory)
    essays = []
    for filepath in sorted(essay_dir.glob("*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                essays.append(text)
    return essays