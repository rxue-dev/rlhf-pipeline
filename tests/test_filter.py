"""Unit tests for filter.py — validates each filtering heuristic independently."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.filter import extract_last_response, filter_rows, is_junk, is_toxic


def make_row(chosen_response: str, rejected_response: str) -> dict:
    """Build a minimal HH-RLHF row with the standard conversation format."""
    return {
        "chosen": f"Human: test prompt\n\nAssistant: {chosen_response}",
        "rejected": f"Human: test prompt\n\nAssistant: {rejected_response}",
    }


class TestExtractLastResponse:
    def test_single_turn(self):
        conv = "Human: hi\n\nAssistant: hello there"
        assert extract_last_response(conv) == "hello there"

    def test_multi_turn(self):
        conv = "Human: hi\n\nAssistant: hey\n\nHuman: how are you\n\nAssistant: doing well"
        assert extract_last_response(conv) == "doing well"

    def test_no_assistant_marker(self):
        assert extract_last_response("just some text") == "just some text"


class TestIsJunk:
    def test_dots_only(self):
        assert is_junk("...")
        assert is_junk(". . . .")

    def test_dont_know_loop(self):
        assert is_junk("I don't know.")
        assert is_junk("i dont know")

    def test_placeholder(self):
        assert is_junk("N/A")
        assert is_junk("none")
        assert is_junk("null")

    def test_normal_text_not_junk(self):
        assert not is_junk("The capital of France is Paris.")
        assert not is_junk("I don't know much about quantum physics, but here's what I can share.")


class TestIsToxic:
    def test_toxic_phrases(self):
        assert is_toxic("you should kill yourself")
        assert is_toxic("You Deserve To Die for that")

    def test_clean_text(self):
        assert not is_toxic("Here's how to solve that problem.")


class TestFilterRows:
    def test_keeps_valid_pair(self):
        rows = [make_row("A" * 60, "B" * 60)]
        result = filter_rows(rows, min_response_length=50)
        assert len(result) == 1

    def test_drops_duplicate_responses(self):
        rows = [make_row("same response here" * 5, "same response here" * 5)]
        result = filter_rows(rows, min_response_length=10)
        assert len(result) == 0

    def test_drops_short_chosen(self):
        rows = [make_row("short", "B" * 60)]
        result = filter_rows(rows, min_response_length=50)
        assert len(result) == 0

    def test_drops_short_rejected(self):
        rows = [make_row("A" * 60, "short")]
        result = filter_rows(rows, min_response_length=50)
        assert len(result) == 0

    def test_drops_junk(self):
        rows = [make_row("..." , "B" * 60)]
        result = filter_rows(rows, min_response_length=1)
        assert len(result) == 0

    def test_drops_toxic(self):
        rows = [make_row("kill yourself " + "x" * 60, "B" * 60)]
        result = filter_rows(rows, min_response_length=10)
        assert len(result) == 0

    def test_min_response_length_configurable(self):
        rows = [make_row("A" * 20, "B" * 20)]
        assert len(filter_rows(rows, min_response_length=10)) == 1
        assert len(filter_rows(rows, min_response_length=30)) == 0
