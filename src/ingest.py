"""Download and cache Anthropic's HH-RLHF dataset from Hugging Face."""

import argparse
from typing import Dict, List, Optional

from datasets import load_dataset


def ingest(split: str = "train", max_samples: Optional[int] = None) -> List[Dict]:
    """Load the HH-RLHF dataset and return raw rows as dicts.

    The dataset has two columns: 'chosen' and 'rejected', each containing
    a full multi-turn conversation string. We load a single split and
    optionally cap the number of rows for faster iteration.
    """
    dataset = load_dataset("Anthropic/hh-rlhf", split=split)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    rows = list(dataset)

    print_stats(split, rows, dataset)
    return rows


def print_stats(split: str, rows: List[Dict], dataset) -> None:
    print(f"\n--- Ingest Stats ---")
    print(f"Split:        {split}")
    print(f"Total rows:   {len(rows)}")
    print(f"Schema:       {list(dataset.features.keys())}")
    print(f"Sample keys:  {list(rows[0].keys()) if rows else '(empty)'}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest HH-RLHF dataset")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max rows to load")
    args = parser.parse_args()

    ingest(split=args.split, max_samples=args.max_samples)
