# RLHF Data Preprocessing Pipeline

A Python pipeline that downloads Anthropic's HH-RLHF dataset from Hugging Face, filters out low-quality preference pairs using configurable heuristics, and reformats the data into training-ready JSONL compatible with TRL's RewardTrainer and SFTTrainer. Built to make every preprocessing decision visible and auditable — this is a learning tool for understanding how raw human preference data becomes model training data.

> **Part of a two-repo project.** This repo handles the *data pipeline* side of RLHF. The companion repo [rlhf-annotator](https://github.com/rxue-dev/rlhf-annotator) is a full-stack annotation interface for collecting pairwise human preferences — the same kind of data this pipeline processes.

## Pipeline Diagram

```
 HuggingFace (Anthropic/hh-rlhf)
        │
        ▼
 ┌─────────────┐
 │  ingest.py   │  Download & cache dataset, select split, cap sample count
 └──────┬──────┘
        │ list[dict] with 'chosen' and 'rejected' conversation strings
        ▼
 ┌─────────────┐
 │  filter.py   │  Remove duplicates, short responses, junk, toxic content
 └──────┬──────┘
        │ list[dict] — quality-filtered subset
        ▼
 ┌──────────────┐
 │ reformat.py   │  Extract prompt/completion, add metadata, write JSONL
 └──────┬───────┘
        │
        ▼
 ┌──────────────┐
 │  output.jsonl  │  One JSON object per line, training-ready
 └──────────────┘
        │
        ▼
 ┌─────────────┐
 │  stats.py    │  Summary report: counts, lengths, percentiles
 └─────────────┘
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (from the rlhf-pipeline directory)
python -m src.pipeline \
  --split train \
  --max_samples 5000 \
  --min_response_length 50 \
  --output output/hh_rlhf_processed.jsonl

# Run stats separately on an existing output file
python -m src.stats output/hh_rlhf_processed.jsonl
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--split` | `train` | Dataset split (`train` or `test`) |
| `--max_samples` | all | Cap the number of rows loaded |
| `--min_response_length` | `50` | Minimum characters for both chosen and rejected |
| `--output` | `output/hh_rlhf_processed.jsonl` | Output file path |

## Output Format

Each line in the output JSONL:

```json
{
  "prompt": "Human: What is...\n\nAssistant:",
  "chosen": "The answer is...",
  "rejected": "I think maybe...",
  "source": "hh-rlhf",
  "split": "train",
  "chosen_length": 142,
  "rejected_length": 98
}
```

## Output Results
<img width="535" height="971" alt="Screenshot 2026-05-29 at 3 00 13 PM" src="https://github.com/user-attachments/assets/8bf659b9-b999-48d8-a0b7-f9eab057f851" />

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

Tests cover each filtering heuristic independently — duplicates, length thresholds, junk patterns, toxicity detection — without needing to download any data.

## Key Design Decisions

### Why these specific filter heuristics?

The filters target failure modes that are common in the HH-RLHF dataset and would degrade training quality:

- **Duplicate responses** (chosen == rejected): These carry zero preference signal — the model learns nothing from a pair where both options are identical.
- **Minimum response length**: Very short responses (under 50 chars) are typically incomplete thoughts, acknowledgments ("OK"), or failed generations that don't contain enough content for meaningful preference learning.
- **Junk patterns** (dots, "N/A", "I don't know" loops): These are artifacts of the data collection process, not genuine assistant responses.
- **Toxic content blocklist**: A short, auditable list of explicit phrases rather than an ML classifier. This keeps the pipeline dependency-free and makes filtering decisions fully transparent — you can read every rule.

We intentionally avoid ML-based toxicity classifiers here because (a) they add heavy dependencies, (b) their decisions are opaque, and (c) for a learning project, explicit rules are easier to reason about and debug.

### Why JSONL over CSV or Parquet?

- **vs CSV**: Preference data contains newlines, commas, and quotes within conversation text. CSV requires escaping that makes files hard to inspect manually. JSONL handles arbitrary string content natively.
- **vs Parquet**: Parquet is more space-efficient and faster for columnar queries, but it's a binary format — you can't `head`, `tail`, or `grep` it. When learning data pipelines, being able to inspect your data with basic Unix tools is more valuable than storage efficiency.
- **TRL compatibility**: TRL's trainers accept JSONL directly, so no conversion step is needed downstream.

### Why preserve length metadata in output?

The `chosen_length` and `rejected_length` fields let downstream consumers filter or stratify by response length without re-parsing text. This is useful for: analyzing whether reward models have length bias, creating length-balanced training batches, and quick sanity checks on data quality.

### Why split the pipeline into discrete steps vs one script?

Each step (ingest → filter → reformat) has a single responsibility and a clean interface (list of dicts in, list of dicts out). This means:

- **Testability**: You can unit test filtering logic without downloading any data.
- **Debuggability**: If output looks wrong, you can run each step independently to find where the problem is.
- **Reusability**: Swap in a different filter without touching ingestion, or reformat the same filtered data into multiple output formats.
- **Readability**: Each file is under 200 lines and does one thing. A monolithic script would mix I/O, business logic, and formatting in ways that obscure what each decision does.
