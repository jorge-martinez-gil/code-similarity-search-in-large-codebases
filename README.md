# Evaluation of Code Similarity Search Strategies in Large-Scale Codebases

[![DOI](https://img.shields.io/badge/DOI-10.1007%2F978--3--662--70140--9__4-blue)](https://doi.org/10.1007/978-3-662-70140-9_4)
[![Springer](https://img.shields.io/badge/Published%20in-TLDKS%20LVII-orange)](https://link.springer.com/chapter/10.1007/978-3-662-70140-9_4)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![GitHub Stars](https://img.shields.io/github/stars/jorge-martinez-gil/code-similarity-search-in-large-codebases)](https://github.com/jorge-martinez-gil/code-similarity-search-in-large-codebases/stargazers)
[![CI](https://github.com/jorge-martinez-gil/code-similarity-search-in-large-codebases/actions/workflows/ci.yml/badge.svg)](https://github.com/jorge-martinez-gil/code-similarity-search-in-large-codebases/actions/workflows/ci.yml)

> **TL;DR** — We systematically benchmark 6 approximate nearest-neighbour engines (Annoy, Elasticsearch, FAISS, HNSWlib, ScaNN, scikit-learn) paired with both TF-IDF and CodeBERT embeddings on the BigCloneBench dataset. Key finding: **FAISS** offers the best scalability; **CodeBERT** gives the highest semantic accuracy; **Elasticsearch** leads on raw query speed.

## 📖 Abstract

This repository accompanies the Springer TLDKS LVII paper *Evaluation of Code Similarity Search Strategies in Large-Scale Codebases* and provides reproducible scripts for benchmarking source-code retrieval pipelines at scale. We evaluate lexical and semantic representations (TF-IDF and CodeBERT) combined with multiple ANN/vector-search engines and report indexing cost, query efficiency, and practical trade-offs for large software corpora.

## 🏗️ Architecture

```text
Raw JSONL ──► Vectorization ──► Index Construction ──► Query Benchmarking ──► Results & Plots
                (TF-IDF /           (FAISS / Annoy /        (latency, throughput,
                CodeBERT)           HNSW / SKLNN /           relevance metrics)
                                    Elasticsearch /
                                    ScaNN)
```

## 🛠️ Methods Evaluated

| Method | Type | Description |
| :--- | :--- | :--- |
| **Annoy** | Tree-based | Approximate nearest-neighbor search with tree-based partitioning. |
| **Elasticsearch** | Inverted Index | Vector and text-based search using tunable scoring. |
| **FAISS** | Clustering/Quantization | Facebook AI Similarity Search; efficient high-dimensional search. |
| **HNSW** | Graph-based | Hierarchical Navigable Small World graphs. |
| **ScaNN** | Quantization | Google's Scalable Nearest Neighbors with partitioning. |
| **SKLNN** | Brute/Tree | scikit-learn nearest-neighbor algorithms. |

## 📊 Key Results

| Finding | Summary |
| :--- | :--- |
| Semantic quality | CodeBERT improves semantic relevance versus TF-IDF across backends. |
| Query speed | Elasticsearch provides the best raw query latency in the reported experiments. |
| Scalability | FAISS offers the strongest scalability profile for very large indexes. |
| Simplicity | scikit-learn is a strong baseline for small/medium datasets. |

## 📂 Repository Structure

```text
.
├── .github/workflows/ci.yml            # CI smoke tests and optional pytest
├── all.py                              # End-to-end benchmark + plots
├── indexing.py                         # Indexing-time benchmark
├── performance.py                      # Latency/QPS benchmark
├── plots.py                            # Benchmark plotting utility
├── testcodebert.py                     # TF-IDF vs CodeBERT benchmark
├── examples/quickstart.py              # Self-contained quickstart demo
├── src/similarity_search/utils.py      # Shared benchmark/data helpers
├── tests/test_utils.py                 # Unit tests for shared helpers
├── docs/ARCHITECTURE.md                # System architecture details
├── docs/RESULTS.md                     # Reproducible results summary
├── requirements.txt                    # Reproducible runtime dependencies
├── pyproject.toml                      # Modern packaging metadata
├── CONTRIBUTING.md                     # Contribution guide
└── CHANGELOG.md                        # Release history
```

## ⚡ Quick Start

```bash
python -m pip install -r requirements.txt
python examples/quickstart.py
```

## 🚀 Full Reproducibility Guide

### Prerequisites

- Python 3.8+
- BigCloneBench JSONL data (CodeXGLUE format)
- Optional: Elasticsearch 8.x for Elasticsearch experiments

### Installation

```bash
pip install -e .
# or
pip install -r requirements.txt
```

### Dataset Setup

1. Download BigCloneBench from CodeXGLUE.
2. Place the file at `data/data.jsonl`.
3. Ensure each line has JSON with a `func` field.

### Running Experiments

```bash
python all.py --data data/data.jsonl --k 3 --trials 100
python indexing.py --data data/data.jsonl --sizes 100 1000 10000
python performance.py --data data/data.jsonl --query "for(int i=0;i<n;i++){sum+=i;}"
python plots.py --data data/data.jsonl --time-plot search_time_comparison.tex
python testcodebert.py --data data/data.jsonl --trials 100
```

### Elasticsearch Setup

```bash
docker run -p 9200:9200 -e discovery.type=single-node -e xpack.security.enabled=false docker.elastic.co/elasticsearch/elasticsearch:8.14.0
```

For authenticated setups, copy `.env.example` to `.env` and configure `ES_HOST`, `ES_USER`, and `ES_PASSWORD`.

## 🔬 Extending the Benchmark

1. Add backend-specific index/search functions in benchmark scripts.
2. Reuse `read_code_snippets` and `benchmark_search` from `src/similarity_search/utils.py`.
3. Add reproducible output and update docs/results tables.
4. Include tests for any shared utilities.

## 📝 Citation

```bibtex
@inbook{Martinez-Gil2025,
  author    = {Martinez-Gil, Jorge and Yin, Shaoyi},
  editor    = {Hameurlain, Abdelkader and Tjoa, A. Min},
  title     = {Evaluation of Code Similarity Search Strategies in Large-Scale Codebases},
  booktitle = {Transactions on Large-Scale Data- and Knowledge-Centered Systems LVII},
  year      = {2025},
  publisher = {Springer Berlin Heidelberg},
  address   = {Berlin, Heidelberg},
  pages     = {99--113},
  isbn      = {978-3-662-70140-9},
  doi       = {10.1007/978-3-662-70140-9_4},
  url       = {https://doi.org/10.1007/978-3-662-70140-9_4}
}
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 License

MIT. See [LICENSE](LICENSE).
