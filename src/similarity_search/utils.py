"""Shared utility functions for code similarity search benchmarks."""

from __future__ import annotations

import json
import time
from typing import List, Tuple

from tqdm import tqdm


def read_code_snippets(filepath: str) -> List[str]:
    """Load code snippets from a JSONL file.

    Each line must be a JSON object containing a ``func`` key with the
    raw source code as a string.

    Args:
        filepath: Path to the ``.jsonl`` file.

    Returns:
        List of code-snippet strings in file order.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
    """
    snippets: List[str] = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            snippets.append(json.loads(line)["func"])
    return snippets


def benchmark_search(
    function,
    args: tuple,
    num_trials: int = 100,
    desc: str = "Benchmarking",
) -> Tuple[float, float]:
    """Time a search function over *num_trials* repeated calls.

    Args:
        function: Callable that performs a single search query.
        args: Positional arguments forwarded to *function*.
        num_trials: Number of repetitions.
        desc: Label shown on the tqdm progress bar.

    Returns:
        Tuple of ``(avg_latency_seconds, queries_per_second)``.
    """
    t0 = time.perf_counter()
    for _ in tqdm(range(num_trials), desc=desc, leave=False):
        function(*args)
    elapsed = time.perf_counter() - t0
    return elapsed / num_trials, num_trials / elapsed
