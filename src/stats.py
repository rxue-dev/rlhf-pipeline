"""Read processed JSONL and print a summary report."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_stats(records: List[Dict]) -> Dict:
    chosen_lengths = [r["chosen_length"] for r in records]
    rejected_lengths = [r["rejected_length"] for r in records]

    return {
        "total_pairs": len(records),
        "avg_chosen_length": sum(chosen_lengths) / len(chosen_lengths) if chosen_lengths else 0,
        "avg_rejected_length": sum(rejected_lengths) / len(rejected_lengths) if rejected_lengths else 0,
        "min_chosen_length": min(chosen_lengths) if chosen_lengths else 0,
        "max_chosen_length": max(chosen_lengths) if chosen_lengths else 0,
        "min_rejected_length": min(rejected_lengths) if rejected_lengths else 0,
        "max_rejected_length": max(rejected_lengths) if rejected_lengths else 0,
        "chosen_length_percentiles": percentiles(chosen_lengths),
        "rejected_length_percentiles": percentiles(rejected_lengths),
    }


def percentiles(values: List[int]) -> Dict:
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    return {
        "p25": s[n // 4],
        "p50": s[n // 2],
        "p75": s[3 * n // 4],
        "p95": s[int(n * 0.95)],
    }


def print_report(stats: Dict) -> None:
    print(f"\n{'='*50}")
    print(f"  RLHF Dataset Summary Report")
    print(f"{'='*50}")
    print(f"  Total pairs:              {stats['total_pairs']}")
    print()
    print(f"  Chosen response length:")
    print(f"    Average:                {stats['avg_chosen_length']:.1f}")
    print(f"    Min / Max:              {stats['min_chosen_length']} / {stats['max_chosen_length']}")
    cp = stats["chosen_length_percentiles"]
    print(f"    p25 / p50 / p75 / p95:  {cp['p25']} / {cp['p50']} / {cp['p75']} / {cp['p95']}")
    print()
    print(f"  Rejected response length:")
    print(f"    Average:                {stats['avg_rejected_length']:.1f}")
    print(f"    Min / Max:              {stats['min_rejected_length']} / {stats['max_rejected_length']}")
    rp = stats["rejected_length_percentiles"]
    print(f"    p25 / p50 / p75 / p95:  {rp['p25']} / {rp['p50']} / {rp['p75']} / {rp['p95']}")
    print(f"{'='*50}\n")


def run(input_path: str) -> None:
    path = Path(input_path)
    if not path.exists():
        print(f"Error: {path} does not exist", file=sys.stderr)
        sys.exit(1)

    records = load_jsonl(str(path))
    if not records:
        print("Error: no records found in file", file=sys.stderr)
        sys.exit(1)

    stats = compute_stats(records)
    print_report(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print stats for processed RLHF JSONL")
    parser.add_argument("input", help="Path to the processed JSONL file")
    args = parser.parse_args()

    run(args.input)
