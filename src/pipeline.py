"""Orchestrate the full RLHF data preprocessing pipeline."""

import argparse
import sys
import time

from src.ingest import ingest
from src.filter import filter_rows
from src.reformat import reformat_and_write
from src.stats import run as run_stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end RLHF data preprocessing pipeline. "
        "Downloads Anthropic's HH-RLHF dataset, filters for quality, "
        "and outputs training-ready JSONL."
    )
    parser.add_argument(
        "--split", default="train",
        help="Dataset split to process (default: train)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum number of samples to load (default: all)"
    )
    parser.add_argument(
        "--min_response_length", type=int, default=50,
        help="Minimum character length for both chosen and rejected responses (default: 50)"
    )
    parser.add_argument(
        "--output", default="output/hh_rlhf_processed.jsonl",
        help="Output JSONL file path (default: output/hh_rlhf_processed.jsonl)"
    )
    args = parser.parse_args()

    start = time.time()

    # Why: The pipeline is split into discrete steps (ingest → filter → reformat)
    # rather than one monolithic script so that each stage can be tested, debugged,
    # and reused independently. You can swap in a different filter without touching
    # ingestion, or reformat the same filtered data into multiple output formats.
    print("=" * 50)
    print("  RLHF Data Preprocessing Pipeline")
    print("=" * 50)

    print("\n[1/4] Ingesting dataset...")
    rows = ingest(split=args.split, max_samples=args.max_samples)

    if not rows:
        print("Error: no data ingested. Check split name and network connection.", file=sys.stderr)
        sys.exit(1)

    print("[2/4] Filtering for quality...")
    filtered = filter_rows(rows, min_response_length=args.min_response_length)

    if not filtered:
        print("Error: all rows filtered out. Try lowering --min_response_length.", file=sys.stderr)
        sys.exit(1)

    print("[3/4] Reformatting to JSONL...")
    reformat_and_write(filtered, output_path=args.output, split=args.split)

    print("[4/4] Generating summary report...")
    run_stats(args.output)

    elapsed = time.time() - start
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
