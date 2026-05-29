"""Convert filtered rows into training-ready JSONL format."""

import json
from pathlib import Path
from typing import Dict, List

from src.filter import extract_last_response


def reformat_and_write(
    rows: List[Dict],
    output_path: str,
    split: str = "train",
) -> List[Dict]:
    """Transform raw HH-RLHF rows into the TRL-compatible training format.

    Each output line is a self-contained JSON object with the prompt (all turns
    up to the final Assistant marker), the chosen and rejected completions,
    and metadata needed downstream.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    formatted = []
    for row in rows:
        prompt = extract_prompt(row["chosen"])
        chosen = extract_last_response(row["chosen"])
        rejected = extract_last_response(row["rejected"])

        record = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "source": "hh-rlhf",
            "split": split,
            # Why: Length metadata lets downstream consumers filter or stratify
            # by response length without re-parsing the text. TRL's RewardTrainer
            # doesn't require these fields, but they're useful for analysis.
            "chosen_length": len(chosen),
            "rejected_length": len(rejected),
        }
        formatted.append(record)

    # Why: JSONL (one JSON object per line) over CSV because preference data
    # contains newlines and commas within conversation text. JSONL handles
    # arbitrary string content without escaping issues. Over Parquet because
    # JSONL is human-readable and debuggable with head/tail/grep — important
    # when you're learning the data pipeline and need to inspect intermediate output.
    with open(dest, "w") as f:
        for record in formatted:
            f.write(json.dumps(record) + "\n")

    print(f"\n--- Reformat Stats ---")
    print(f"Records written: {len(formatted)}")
    print(f"Output file:     {dest}")
    print()

    return formatted


def extract_prompt(conversation: str) -> str:
    """Extract everything up to (and including) the final 'Assistant:' marker.

    This becomes the prompt that a model would condition on during training.
    The completion (chosen or rejected) is what follows.
    """
    # Why: We find the last occurrence of "\n\nAssistant:" and include it,
    # because the training format expects the prompt to end with the assistant
    # turn marker so the model learns to generate from that point.
    last_marker = conversation.rfind("\n\nAssistant:")
    if last_marker == -1:
        return conversation.strip()
    return conversation[:last_marker + len("\n\nAssistant:")].strip()
