
# Evaluation of Code Similarity Search Strategies in Large-Scale Codebases

**Authors:**  
Jorge Martinez-Gil, Software Competence Center Hagenberg GmbH  
Shaoyi Yin, Paul Sabatier University, IRIT Laboratory

**Published at:**  
Transactions on Large-Scale Data- and Knowledge-Centered Systems LVII](https://link.springer.com/chapter/10.1007/978-3-662-70140-9_4)  
Springer

---

## Overview

Automatically identifying similar code fragments in large repositories is essential for code reuse, debugging, and software maintenance. While multiple code similarity search solutions exist, systematic comparisons remain limited. This work presents an empirical evaluation of both classical and emerging code similarity search techniques, focusing on practical performance in real-world large-scale codebases.

## Main Contributions

- Comparative analysis of traditional and modern techniques for code similarity search.
- Benchmarking of popular methods, including Annoy, Elasticsearch, FAISS, HNSW, ScaNN, and Scikit-learn Nearest Neighbors (SKLNN).
- Evaluation across metrics such as indexing time, search speed, and semantic relevance.
- Guidance on selecting suitable strategies for different scenarios.

## Methods Evaluated

- **Annoy:** Approximate nearest-neighbor search with tree-based partitioning.
- **Elasticsearch:** Vector and text-based search using tunable scoring.
- **FAISS:** Efficient high-dimensional similarity search with multiple indexing options.
- **HNSW:** Hierarchical graph-based nearest-neighbor search.
- **ScaNN:** Scalable search with quantization and partitioning.
- **SKLNN:** General-purpose nearest neighbors from Scikit-learn.

Vectorization approaches include both TF-IDF and CodeBERT.

## Dataset

Experiments use the [BigCloneBench](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench) dataset, featuring real-world and student project code fragments from multiple programming languages.

## Key Findings

- **Accuracy:** CodeBERT achieves the highest accuracy, remaining stable as dataset size increases.
- **Performance:** Elasticsearch provides the fastest query times, though at the cost of slower indexing.
- **Scalability:** FAISS demonstrates strong long-term scalability; SKLNN performs well for smaller datasets.
- **Suitability:** All methods are viable for integration, depending on requirements and scale.

## Citation

If you use this work, please cite:

```bibtex
@inbook{Martinez-Gil2025,
	author="Martinez-Gil, Jorge and Yin, Shaoyi",
	editor="Hameurlain, Abdelkader and Tjoa, A. Min",
	title="Evaluation of Code Similarity Search Strategies in Large-Scale Codebases",
	bookTitle="Transactions on Large-Scale Data- and Knowledge-Centered Systems LVII",
	year="2025",
	publisher="Springer Berlin Heidelberg",
	address="Berlin, Heidelberg",
	pages="99--113",
	isbn="978-3-662-70140-9",
	doi="10.1007/978-3-662-70140-9_4",
	url="https://doi.org/10.1007/978-3-662-70140-9_4"
}
```
