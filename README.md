
# Evaluation of Code Similarity Search Strategies in Large-Scale Codebases

[![DOI](https://img.shields.io/badge/DOI-10.1007%2F978--3--662--70140--9__4-blue)](https://doi.org/10.1007/978-3-662-70140-9_4)
[![Springer](https://img.shields.io/badge/Published%20in-TLDKS%20LVII-orange)](https://link.springer.com/chapter/10.1007/978-3-662-70140-9_4)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE) 

**Authors:** [Jorge Martinez-Gil](https://www.scch.at/) (Software Competence Center Hagenberg GmbH) & [Shaoyi Yin](https://www.irit.fr/) (Paul Sabatier University, IRIT Laboratory)

---

## 📖 Overview

Automatically identifying similar code fragments in large repositories is essential for **code reuse**, **debugging**, and **software maintenance**. While multiple code similarity search solutions exist, systematic comparisons remain limited. 

This repository contains the source code and benchmark results for our paper, **"Evaluation of Code Similarity Search Strategies in Large-Scale Codebases"**. We present an empirical evaluation of both classical and emerging code similarity search techniques, focusing on practical performance in real-world large-scale codebases using **BigCloneBench**.

> **Read the full paper:** [Transactions on Large-Scale Data- and Knowledge-Centered Systems LVII (Springer)](https://link.springer.com/chapter/10.1007/978-3-662-70140-9_4)

## 🚀 Key Contributions

* **Systematic Benchmarking:** A comparative analysis of traditional (TF-IDF) vs. modern (CodeBERT) vectorization techniques.
* **Algorithm Evaluation:** Performance metrics for 6 popular similarity search methods (Annoy, FAISS, HNSW, etc.).
* **Multi-Metric Analysis:** Evaluation across **indexing time**, **search speed**, and **semantic relevance**.
* **Scalability Insights:** Guidance on selecting strategies for massive codebases.

## 🛠 Methods Evaluated

We evaluated the following strategies combined with **TF-IDF** and **CodeBERT** vectorization:

| Method | Type | Description |
| :--- | :--- | :--- |
| **Annoy** | Tree-based | Approximate nearest-neighbor search with tree-based partitioning. |
| **Elasticsearch** | Inverted Index | Vector and text-based search using tunable scoring. |
| **FAISS** | Clustering/Quantization | Facebook AI Similarity Search; efficient high-dimensional search. |
| **HNSW** | Graph-based | Hierarchical Navigable Small World graphs. |
| **ScaNN** | Quantization | Google's Scalable Nearest Neighbors with partitioning. |
| **SKLNN** | Brute/Tree | Scikit-learn's general-purpose nearest neighbor algorithms. |

## 📊 Key Findings

Our experiments on the [BigCloneBench](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench) dataset revealed:

1.  **Accuracy:** **CodeBERT** achieves the highest semantic accuracy and remains stable as dataset size increases.
2.  **Query Speed:** **Elasticsearch** provides the fastest query times, though with higher indexing overhead.
3.  **Scalability:** **FAISS** demonstrates the best long-term scalability for massive datasets.
4.  **Small Data:** **SKLNN** is highly effective and simple for smaller datasets but struggles to scale.


## 📂 Dataset

This project utilizes the **BigCloneBench** dataset, a standard benchmark for code clone detection.

-   **Source:** [CodeXGLUE / BigCloneBench](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench)  

## 📝 Citation

If you use this code or our findings in your research, please cite the following paper:

Fragmento de código

```
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
  url       = {[https://doi.org/10.1007/978-3-662-70140-9_4](https://doi.org/10.1007/978-3-662-70140-9_4)}
}

```
