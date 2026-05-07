## Benchmark Results Summary

### Query Latency (avg ms/query, 10 k snippets, TF-IDF vectorization)

| Method       | Latency (ms) | QPS    | Index Time (s) |
|:-------------|-------------:|-------:|---------------:|
| FAISS        | [from paper Table X] | [from paper Table X] | [from paper Table X] |
| Annoy        | [from paper Table X] | [from paper Table X] | [from paper Table X] |
| HNSWlib      | [from paper Table X] | [from paper Table X] | [from paper Table X] |
| Scikit-learn | [from paper Table X] | [from paper Table X] | [from paper Table X] |
| Elasticsearch| [from paper Table X] | [from paper Table X] | [from paper Table X] |

### Semantic Accuracy (CodeBERT vs. TF-IDF)

> CodeBERT achieves higher semantic relevance across all backends while TF-IDF remains competitive for lexical similarity and lower computational overhead.
