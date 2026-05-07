"""Compare TF-IDF and CodeBERT vectorization with ANN search backends."""

import argparse
import os

import faiss
import hnswlib
import matplotlib.pyplot as plt
import numpy as np
from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModel, AutoTokenizer

from src.similarity_search.utils import benchmark_search, read_code_snippets

try:
    import tikzplotlib
except ImportError:  # pragma: no cover - environment-dependent optional dependency
    tikzplotlib = None


def vectorize_snippets_tfidf(code_snippets):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(code_snippets).toarray().astype(np.float32)
    return tfidf_matrix, vectorizer


def vectorize_snippets_codebert(code_snippets):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    embeddings = []
    for snippet in code_snippets:
        inputs = tokenizer(snippet, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy()[0])
    return np.array(embeddings, dtype=np.float32)


def faiss_search(tfidf_matrix, query_vector, k):
    index = faiss.IndexFlatL2(tfidf_matrix.shape[1])
    index.add(tfidf_matrix.astype(np.float32))
    _, indices = index.search(query_vector.astype(np.float32), k)
    return indices


def annoy_search(tfidf_matrix, query_vector, k, dimension):
    index = AnnoyIndex(dimension, "angular")
    for i, vector in enumerate(tfidf_matrix):
        index.add_item(i, vector)
    index.build(10)
    return index.get_nns_by_vector(query_vector[0], k, include_distances=False)


def hnsw_search(tfidf_matrix, query_vector, k, space="l2", dimension=None):
    dimension = dimension or tfidf_matrix.shape[1]
    index = hnswlib.Index(space=space, dim=dimension)
    index.init_index(max_elements=len(tfidf_matrix), ef_construction=200, M=16)
    index.add_items(tfidf_matrix)
    index.set_ef(200)
    labels, _ = index.knn_query(query_vector, k)
    return labels


def sklearn_nn_search(tfidf_matrix, query_vector, k):
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(tfidf_matrix)
    _, indices = nn.kneighbors(query_vector)
    return indices


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.path.join("data", "data.jsonl"))
    parser.add_argument("--query", default="int result = 1; for (int i = 1; i <= n; i++) { result *= i; }")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--time-plot", default="search_time_comparison.tex")
    parser.add_argument("--throughput-plot", default="search_throughput_comparison.tex")
    return parser.parse_args()


def main():
    args = parse_args()
    code_snippets = read_code_snippets(args.data)

    tfidf_matrix, tfidf_vectorizer = vectorize_snippets_tfidf(code_snippets)
    query_vector_tfidf = tfidf_vectorizer.transform([args.query]).toarray().astype(np.float32)

    _ = vectorize_snippets_codebert(code_snippets)
    _ = vectorize_snippets_codebert([args.query])

    dimension = tfidf_matrix.shape[1]
    methods = [
        ("FAISS", faiss_search, (tfidf_matrix, query_vector_tfidf, args.k)),
        ("SklearnNN", sklearn_nn_search, (tfidf_matrix, query_vector_tfidf, args.k)),
        ("Annoy", annoy_search, (tfidf_matrix, query_vector_tfidf, args.k, dimension)),
        ("HNSW", hnsw_search, (tfidf_matrix, query_vector_tfidf, args.k, "l2", dimension)),
    ]

    times, throughputs = [], []
    for name, function, method_args in methods:
        avg_time, throughput = benchmark_search(function, method_args, num_trials=args.trials, desc=name)
        times.append(avg_time)
        throughputs.append(throughput)
        print(f"{name}: {avg_time:.4f} sec/query, {throughput:.2f} queries/sec")

    plt.figure(figsize=(10, 5))
    plt.bar([method[0] for method in methods], times, color="blue")
    plt.xlabel("Search Method")
    plt.ylabel("Average Time per Query (s)")
    plt.title("Search Performance Comparison")
    if tikzplotlib:
        tikzplotlib.save(args.time_plot)
    else:
        plt.savefig(args.time_plot.replace(".tex", ".png"))

    plt.figure(figsize=(10, 5))
    plt.bar([method[0] for method in methods], throughputs, color="green")
    plt.xlabel("Search Method")
    plt.ylabel("Queries per Second")
    plt.title("Search Throughput Comparison")
    if tikzplotlib:
        tikzplotlib.save(args.throughput_plot)
    else:
        plt.savefig(args.throughput_plot.replace(".tex", ".png"))


if __name__ == "__main__":
    main()
