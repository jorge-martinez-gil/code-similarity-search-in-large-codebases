"""Unit tests for shared utility functions."""

import json
import os
import tempfile

import pytest


def test_read_code_snippets_basic():
    from src.similarity_search.utils import read_code_snippets

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as handle:
        for snippet in ["def foo(): pass", "def bar(): return 1"]:
            handle.write(json.dumps({"func": snippet}) + "\n")
        path = handle.name
    try:
        snippets = read_code_snippets(path)
        assert snippets == ["def foo(): pass", "def bar(): return 1"]
    finally:
        os.unlink(path)


def test_read_code_snippets_missing_file():
    from src.similarity_search.utils import read_code_snippets

    with pytest.raises(FileNotFoundError):
        read_code_snippets("/nonexistent/path/data.jsonl")


def test_benchmark_search_returns_valid_metrics():
    from src.similarity_search.utils import benchmark_search

    call_count = []

    def dummy_search(x):
        call_count.append(1)
        return x

    avg_time, qps = benchmark_search(dummy_search, (42,), num_trials=10)
    assert avg_time > 0
    assert qps > 0
    assert len(call_count) == 10
