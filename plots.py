"""
Benchmark and Visualize Search Latency and Throughput for Vector Search Libraries

This script benchmarks query time and throughput for various code snippet vector search libraries
(FAISS, Annoy, HNSWlib, scikit-learn, Elasticsearch), and plots the results using matplotlib and tikzplotlib.

Expected input: JSONL file with one object per line, containing a 'func' field with code.

Dependencies:
    - numpy
    - sklearn
    - faiss
    - annoy
    - hnswlib
    - elasticsearch
    - matplotlib
    - tikzplotlib
    - tqdm
    - json
    - time

Author: Jorge Martinez-Gil
"""

import json
import time
import matplotlib.pyplot as plt
import tikzplotlib
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

def benchmark_search(function, args, num_trials=100):
    """
    Measure average query latency and throughput for a search function.

    Args:
        function (callable): The search function to benchmark.
        args (tuple): Arguments to pass to the function.
        num_trials (int): Number of repeated queries.

    Returns:
        tuple: (average_time_per_query, queries_per_second)
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
    Search for nearest neighbors using FAISS (exact L2 search).

    Args:
        tfidf_matrix (np.ndarray): Matrix with shape (n_samples, n_features).
        query_vector (np.ndarray): Query vector, shape (1, n_features).
        k (int): Number of neighbors to retrieve.

    Returns:
        np.ndarray: Indices of top-k matches.
    """
    dimension = tfidf_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(tfidf_matrix.astype(np.float32))
    _, I = index.search(query_vector, k)
    return I

def annoy_search(tfidf_matrix , query_vector, k, dimension):
    """
    Search for nearest neighbors using Annoy (approximate angular distance).

    Args:
        tfidf_matrix (np.ndarray): Matrix with shape (n_samples, n_features).
        query_vector (np.ndarray): Query vector, shape (1, n_features).
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
    Search for nearest neighbors using HNSWlib (approximate search).

    Args:
        tfidf_matrix (np.ndarray): Matrix with shape (n_samples, n_features).
        query_vector (np.ndarray): Query vector, shape (1, n_features).
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
    Search for nearest neighbors using scikit-learn's NearestNeighbors (exact search).

    Args:
        tfidf_matrix (np.ndarray): Matrix with shape (n_samples, n_features).
        query_vector (np.ndarray): Query vector, shape (1, n_features).
        k (int): Number of neighbors to retrieve.

    Returns:
        np.ndarray: Indices of top-k matches.
    """
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(tfidf_matrix)
    distances, indices = nn.kneighbors(query_vector)
    return indices

def elastic_search(es, index_name, query_vector, k):
    """
    Search for similar code snippets using Elasticsearch dense_vector field.

    Args:
        es (Elasticsearch): Authenticated Elasticsearch client.
        index_name (str): Elasticsearch index name.
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
    - Benchmarks each search method.
    - Plots and saves results as TikZ for LaTeX.
    """
    filename = "data\data.jsonl"
    query_snippet = "int result = 1; for (int i = 1; i <= n; i++) { result *= i; }"
    code_snippets = read_code_snippets(filename)
    tfidf_matrix, vectorizer = vectorize_snippets(code_snippets)
    query_vector = vectorizer.transform([query_snippet]).toarray().astype(np.float32)

    es = Elasticsearch(
        ["http://localhost:9200"],
        basic_auth=('elastic', 'nGRuZaLMB0zgEHHLFLzz')
    )

    methods = [
        ("FAISS", faiss_search, (tfidf_matrix, query_vector, 3)),
        ("Annoy", annoy_search, (tfidf_matrix, query_vector, 3, tfidf_matrix.shape[1])),
        ("HNSWlib", hnsw_search, (tfidf_matrix, query_vector, 3)),
        ("Scikit-learn", sklearn_search, (tfidf_matrix, query_vector, 3))
    ]

    # Prepare query for Elasticsearch: reduce and normalize vector dimensionality
    svd = TruncatedSVD(n_components=4096)
    svd.fit(tfidf_matrix)
    normalized_query_vector = normalize(svd.transform(vectorizer.transform([query_snippet]).toarray()), norm='l2')
    methods.append(("Elastic", elastic_search, (es, "code_snippets_index99", normalized_query_vector[0], 3)))

    times = []
    throughputs = []

    for name, function, args in methods:
        avg_time, throughput = benchmark_search(function, args)
        times.append(avg_time)
        throughputs.append(throughput)
        print(f"{name}: {avg_time:.4f} sec/query, {throughput:.2f} queries/sec")

    # Plotting: Average Time per Query
    plt.figure(figsize=(10, 5))
    plt.bar([method[0] for method in methods], times, color='blue')
    plt.xlabel('Search Method')
    plt.ylabel('Average Time per Query (s)')
    plt.title('Search Performance Comparison')
    tikzplotlib.save("search_time_comparison.tex")

    # Plotting: Throughput
    plt.figure(figsize=(10, 5))
    plt.bar([method[0] for method in methods], throughputs, color='green')
    plt.xlabel('Search Method')
    plt.ylabel('Queries per Second')
    plt.title('Search Throughput Comparison')
    tikzplotlib.save("search_throughput_comparison.tex")

if __name__ == "__main__":
    main()
