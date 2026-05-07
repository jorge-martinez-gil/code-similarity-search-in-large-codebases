# Architecture

## Pipeline Overview

```text
Raw JSONL ──► Vectorization ──► Index Construction ──► Query Benchmarking ──► Results & Plots
                (TF-IDF /           (FAISS / Annoy /        (latency, throughput,
                CodeBERT)           HNSW / SKLNN /           relevance metrics)
                                    Elasticsearch /
                                    ScaNN)
```

## Components

- **Data ingestion** (`src/similarity_search/utils.py`, all scripts): reads `data/data.jsonl` and extracts `func`.
- **Vectorization** (`all.py`, `performance.py`, `plots.py`, `testcodebert.py`): TF-IDF and CodeBERT embeddings.
- **Index construction** (`indexing.py`): measures backend index build times over increasing corpus sizes.
- **Query benchmarking** (`all.py`, `performance.py`, `plots.py`, `testcodebert.py`): repeated top-k search for latency/QPS.
- **Artifacts** (`plots.py`, `all.py`, `testcodebert.py`): exports TikZ plot files for publication.
