"""
Benchmarking Script for Code Snippet Search Methods

This script compares various search libraries and frameworks for retrieving the most similar code snippets
using TF-IDF vectorization. It supports FAISS, Annoy, HNSWlib, scikit-learn, and Elasticsearch.

Expected input: JSONL file with one object per line, where each object contains a "func" field with code.

Dependencies:
    - numpy
    - sklearn
    - faiss
    - annoy
    - hnswlib
    - elasticsearch
    - tqdm
    - json
    - time

Author: Jorge Martinez-Gil
"""

import json
import time
from elasticsearch import Elasticsearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
import numpy as np
import faiss
from annoy import AnnoyIndex
import hnswlib
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

def benchmark_search(function, args, num_trials=5):
    """
    Run a search function multiple times and measure average latency and throughput.

    Args:
        function (callable): The search function to benchmark.
        args (tuple): Arguments to pass to the function.
        num_trials (int): Number of trials to run.

    Returns:
        tuple: (average time per query, queries per second)
    """
    start_time = time.time()
    for _ in tqdm(range(num_trials), desc='Processing searches'):
        results = function(*args)
    elapsed_time = time.time() - start_time
    throughput = num_trials / elapsed_time
    return elapsed_time / num_trials, throughput

def read_code_snippets(filename):
    """
    Load code snippets from a JSONL file. Each line must contain a 'func' field.

    Args:
        filename (str): Path to the JSONL file.

    Returns:
        list: List of code snippet strings.
    """
    code_snippets = []
    with open(filename, 'r') as file:
        for line in file:
            data = json.loads(line)
            code_snippet = data["func"]
            code_snippets.append(code_snippet)
    return code_snippets

def vectorize_snippets(code_snippets):
    """
    Transform code snippets into TF-IDF vectors.

    Args:
        code_snippets (list): List of code snippet strings.

    Returns:
        tuple: (tfidf_matrix, vectorizer)
            tfidf_matrix: np.ndarray with shape (n_samples, n_features)
            vectorizer: fitted TfidfVectorizer instance
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(code_snippets).toarray()
    return tfidf_matrix, vectorizer

def faiss_search(tfidf_matrix, query_vector, k):
    """
    Perform top-k search with FAISS (exact L2 search).

    Args:
        tfidf_matrix (np.ndarray): Matrix with shape (n_samples, n_features).
        query_vector (np.ndarray): Matrix with shape (1, n_features).
        k (int): Number of neighbors to retrieve.

    Returns:
        np.ndarray: Indices of top-k matches.
    """
    dimension = tfidf_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(tfidf_matrix.astype(np.float32))
    _, I = index.search(query_vector, k)
    return I

def annoy_search(tfidf_matrix, query_vector, k, dimension):
    """
    Perform top-k search with Annoy (approximate angular distance).

    Args:
        tfidf_matrix (np.ndarray): Matrix with shape (n_samples, n_features).
        query_vector (np.ndarray): Matrix with shape (1, n_features).
        k (int): Number of neighbors to retrieve.
        dimension (int): Vector dimensionality.

    Returns:
        list: Indices of top-k matches.
    """
    index = AnnoyIndex(dimension, 'angular')
    for i, vector in enumerate(tfidf_matrix):
        index.add_item(i, vector)
    index.build(10)
    I = index.get_nns_by_vector(query_vector[0], k, include_distances=False)
    return I

def hnsw_search(tfidf_matrix, query_vector, k, space='l2', dimension=None):
    """
    Perform top-k search with HNSWlib (approximate search).

    Args:
        tfidf_matrix (np.ndarray): Matrix with shape (n_samples, n_features).
        query_vector (np.ndarray): Matrix with shape (1, n_features).
        k (int): Number of neighbors to retrieve.
        space (str): 'l2' for Euclidean, 'cosine' for cosine similarity.
        dimension (int): Vector dimensionality.

    Returns:
        np.ndarray: Indices of top-k matches.
    """
    dimension = dimension or tfidf_matrix.shape[1]
    p = hnswlib.Index(space=space, dim=dimension)
    p.init_index(max_elements=len(tfidf_matrix), ef_construction=200, M=16)
    p.add_items(tfidf_matrix)
    p.set_ef(200)  # ef should always be > k
    labels, distances = p.knn_query(query_vector, k)
    return labels

def sklearn_search(tfidf_matrix, query_vector, k):
    """
    Perform top-k search using scikit-learn's NearestNeighbors (exact search).

    Args:
        tfidf_matrix (np.ndarray): Matrix with shape (n_samples, n_features).
        query_vector (np.ndarray): Matrix with shape (1, n_features).
        k (int): Number of neighbors to retrieve.

    Returns:
        np.ndarray: Indices of top-k matches.
    """
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(tfidf_matrix)
    distances, indices = nn.kneighbors(query_vector)
    return indices

def elastic_search(es, index_name, query_vector, k):
    """
    Search for similar code snippets using Elasticsearch vector similarity (cosine similarity).

    Args:
        es (Elasticsearch): Authenticated Elasticsearch client.
        index_name (str): Index name.
        query_vector (np.ndarray): Query vector (1D numpy array).
        k (int): Number of results.

    Returns:
        list: List of snippet strings from Elasticsearch hits.
    """
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                "params": {"query_vector": query_vector.tolist()}
            }
        }
    }
    response = es.search(index=index_name, body={"query": script_query, "size": k})
    return [hit["_source"]["snippet"] for hit in response["hits"]["hits"]]

def main():
    """
    Main entry point:
    - Loads data and query.
    - Vectorizes code snippets.
    - Initializes Elasticsearch client.
    - Benchmarks all search methods.
    - Prints results.
    """
    filename = "data\data.jsonl"
    query_snippet = "int result = 1; for (int i = 1; i <= n; i++) { result *= i; }"
    code_snippets = read_code_snippets(filename)
    tfidf_matrix, vectorizer = vectorize_snippets(code_snippets)
    query_vector = vectorizer.transform([query_snippet]).toarray().astype(np.float32)

    es = Elasticsearch(
        ["http://localhost:9200"],
        basic_auth=('elastic', '---')
    )

    # Benchmark each method
    faiss_time, faiss_throughput = benchmark_search(faiss_search, (tfidf_matrix, query_vector, 3))
    annoy_time, annoy_throughput = benchmark_search(annoy_search, (tfidf_matrix, query_vector, 3, tfidf_matrix.shape[1]))
    hnsw_time, hnsw_throughput = benchmark_search(hnsw_search, (tfidf_matrix, query_vector, 3))
    sklearn_time, sklearn_throughput = benchmark_search(sklearn_search, (tfidf_matrix, query_vector, 3))

    # For Elasticsearch, perform dimensionality reduction and normalization
    svd = TruncatedSVD(n_components=4096)
    svd.fit(tfidf_matrix)
    normalized_query_vector = normalize(svd.transform(vectorizer.transform([query_snippet]).toarray()), norm='l2')
    elastic_time, elastic_throughput = benchmark_search(
        elastic_search,
        (es, "code_snippets_index99", normalized_query_vector[0], 3)
    )

    print(f"FAISS: {faiss_time:.4f} sec/query, {faiss_throughput:.2f} queries/sec")
    print(f"Annoy: {annoy_time:.4f} sec/query, {annoy_throughput:.2f} queries/sec")
    print(f"HNSWlib: {hnsw_time:.4f} sec/query, {hnsw_throughput:.2f} queries/sec")
    print(f"Elastic: {elastic_time:.4f} sec/query, {elastic_throughput:.2f} queries/sec")
    print(f"Scikit-learn NearestNeighbors: {sklearn_time:.4f} sec/query, {sklearn_throughput:.2f} queries/sec")

if __name__ == "__main__":
    main()
