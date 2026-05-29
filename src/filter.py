"""Filter low-quality preference pairs from the HH-RLHF dataset."""

import re
from typing import Dict, List

# Why: These are common filler/junk patterns found in web-scraped dialogue data.
# They don't represent meaningful assistant responses and would add noise to training.
JUNK_PATTERNS = [
    re.compile(r"^(\s*\.+\s*)+$"),           # responses that are just dots/ellipsis
    re.compile(r"(?i)^(i don'?t know\.?\s*)+$"),  # vacuous "I don't know" loops
    re.compile(r"(?i)^(n/?a|none|null)\s*$"),     # placeholder non-answers
]

# Why: Slurs and explicit hate speech in *assistant* responses are not useful for
# training a helpful assistant. We use a short blocklist rather than an ML classifier
# to keep the pipeline dependency-free and fully auditable.
TOXIC_SUBSTRINGS = [
    "kill yourself",
    "you deserve to die",
]


def filter_rows(
    rows: List[Dict],
    min_response_length: int = 50,
) -> List[Dict]:
    """Apply quality filters and return surviving rows plus drop stats."""
    stats = {
        "input_count": len(rows),
        "duplicate_response": 0,
        "chosen_too_short": 0,
        "rejected_too_short": 0,
        "junk_content": 0,
        "toxic_content": 0,
    }

    kept = []
    for row in rows:
        chosen = row["chosen"]
        rejected = row["rejected"]

        chosen_response = extract_last_response(chosen)
        rejected_response = extract_last_response(rejected)

        # Why: If chosen == rejected, the pair carries zero preference signal.
        if chosen_response == rejected_response:
            stats["duplicate_response"] += 1
            continue

        # Why: Very short responses rarely contain substantive content.
        # The threshold is configurable so users can tune quality vs quantity.
        if len(chosen_response) < min_response_length:
            stats["chosen_too_short"] += 1
            continue
        if len(rejected_response) < min_response_length:
            stats["rejected_too_short"] += 1
            continue

        if is_junk(chosen_response) or is_junk(rejected_response):
            stats["junk_content"] += 1
            continue

        if is_toxic(chosen_response) or is_toxic(rejected_response):
            stats["toxic_content"] += 1
            continue

        kept.append(row)

    stats["output_count"] = len(kept)
    stats["total_dropped"] = stats["input_count"] - stats["output_count"]

    print_stats(stats)
    return kept


def extract_last_response(conversation: str) -> str:
    """Pull out the final Assistant turn from a multi-turn conversation string.

    The HH-RLHF format is: "Human: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: ..."
    We want only the last assistant response, since that's what was rated.
    """
    # Why: split on "Assistant:" and take the last segment. This is more robust
    # than regex for the variable formatting in this dataset.
    parts = conversation.split("\n\nAssistant:")
    if len(parts) < 2:
        return conversation.strip()
    return parts[-1].strip()


def is_junk(text: str) -> bool:
    return any(p.match(text) for p in JUNK_PATTERNS)


def is_toxic(text: str) -> bool:
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in TOXIC_SUBSTRINGS)


def print_stats(stats: dict) -> None:
    print(f"\n--- Filter Stats ---")
    print(f"Input pairs:      {stats['input_count']}")
    print(f"Duplicate resp:   {stats['duplicate_response']}")
    print(f"Chosen too short: {stats['chosen_too_short']}")
    print(f"Rejected too short: {stats['rejected_too_short']}")
    print(f"Junk content:     {stats['junk_content']}")
    print(f"Toxic content:    {stats['toxic_content']}")
    print(f"Total dropped:    {stats['total_dropped']}")
    print(f"Output pairs:     {stats['output_count']}")
    drop_rate = stats['total_dropped'] / stats['input_count'] * 100 if stats['input_count'] else 0
    print(f"Drop rate:        {drop_rate:.1f}%")
    print()
