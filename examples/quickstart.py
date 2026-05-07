#!/usr/bin/env python3
"""Quickstart Example — Code Similarity Search.

Demonstrates how to find similar code snippets using TF-IDF + FAISS
without requiring any external dataset.
"""

import time

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

CORPUS = [
    "for i in range(n): result += i",
    "total = 0\nfor i in range(n):\n    total += i",
    "def factorial(n):\n    return 1 if n == 0 else n * factorial(n-1)",
    "def factorial(n):\n    r = 1\n    for i in range(1, n+1):\n        r *= i\n    return r",
    "x = sorted(lst, key=lambda v: v[1])",
    "lst.sort(key=lambda v: v[1])",
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid+1\n        else: hi = mid-1\n    return -1",
    "import bisect\nbisect.bisect_left(arr, target)",
]

QUERY = "for i in range(n): result += i"
K = 3


def main():
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(CORPUS).toarray().astype(np.float32)
    query = vectorizer.transform([QUERY]).toarray().astype(np.float32)

    faiss_index = faiss.IndexFlatL2(matrix.shape[1])
    faiss_index.add(matrix)

    t0 = time.perf_counter()
    _, faiss_indices = faiss_index.search(query, K)
    faiss_ms = (time.perf_counter() - t0) * 1000

    nn = NearestNeighbors(n_neighbors=K, algorithm="auto").fit(matrix)
    t1 = time.perf_counter()
    _, sklearn_indices = nn.kneighbors(query)
    sklearn_ms = (time.perf_counter() - t1) * 1000

    print(f"\nQuery: '{QUERY}'")
    print("─" * 50)
    print(f"FAISS top-{K} results ({faiss_ms:.2f} ms):")
    for rank, idx in enumerate(faiss_indices[0], start=1):
        print(f"  [{rank}] {CORPUS[idx].replace(chr(10), '\\n')[:80]}")

    print(f"\nscikit-learn top-{K} results ({sklearn_ms:.2f} ms):")
    for rank, idx in enumerate(sklearn_indices[0], start=1):
        print(f"  [{rank}] {CORPUS[idx].replace(chr(10), '\\n')[:80]}")


if __name__ == "__main__":
    main()
