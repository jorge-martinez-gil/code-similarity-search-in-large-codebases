"""
Indexing Time Benchmark for Code Snippet Search Engines

This script measures the time needed to build indexes for various vector search libraries (FAISS, Annoy, HNSWlib,
scikit-learn NearestNeighbors, and Elasticsearch) using code snippets vectorized with TF-IDF.

Expected input: JSONL file with one object per line, containing a 'func' field with code.

Dependencies:
    - numpy
    - sklearn
    - faiss
    - annoy
    - hnswlib
    - elasticsearch
    - json
    - time

Author: Jorge Martinez-Gil
"""

import json
import time
from elasticsearch import Elasticsearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import numpy as np
import faiss
from annoy import AnnoyIndex
import hnswlib
from sklearn.neighbors import NearestNeighbors

def read_code_snippets(filename):
    """
    Load code snippets from a JSONL file. Each line must contain a 'func' field.

    Args:
        filename (str): Path to the JSONL file.

    Returns:
        list: List of code snippet strings. Empty list if file not found.
    """
    code_snippets = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                data = json.loads(line)
                code_snippet = data["func"]
                code_snippets.append(code_snippet)
    except FileNotFoundError:
        print(f"File {filename} not found.")
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
    print("Vectorizing code snippets...")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(code_snippets).toarray()
    print("Vectorization complete.")
    return tfidf_matrix, vectorizer

def faiss_index(tfidf_matrix):
    """
    Build a FAISS index from the given TF-IDF matrix.

    Args:
        tfidf_matrix (np.ndarray): Array with shape (n_samples, n_features).

    Returns:
        tuple: (index, build_time_seconds)
    """
    start_time = time.time()
    dimension = tfidf_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(tfidf_matrix.astype(np.float32))
    return index, time.time() - start_time

def annoy_index(tfidf_matrix, dimension):
    """
    Build an Annoy index from the given TF-IDF matrix.

    Args:
        tfidf_matrix (np.ndarray): Array with shape (n_samples, n_features).
        dimension (int): Vector dimensionality.

    Returns:
        tuple: (AnnoyIndex, build_time_seconds)
    """
    start_time = time.time()
    index = AnnoyIndex(dimension, 'angular')
    for i, vector in enumerate(tfidf_matrix):
        index.add_item(i, vector)
    index.build(10)
    return index, time.time() - start_time

def hnsw_index(tfidf_matrix, space='l2'):
    """
    Build a HNSWlib index from the given TF-IDF matrix.

    Args:
        tfidf_matrix (np.ndarray): Array with shape (n_samples, n_features).
        space (str): 'l2' for Euclidean, 'cosine' for cosine similarity.

    Returns:
        tuple: (HNSWlib.Index, build_time_seconds)
    """
    start_time = time.time()
    dimension = tfidf_matrix.shape[1]
    p = hnswlib.Index(space=space, dim=dimension)
    p.init_index(max_elements=len(tfidf_matrix), ef_construction=200, M=16)
    p.add_items(tfidf_matrix)
    return p, time.time() - start_time

def sklearn_index(tfidf_matrix):
    """
    Build a scikit-learn NearestNeighbors index from the given TF-IDF matrix.

    Args:
        tfidf_matrix (np.ndarray): Array with shape (n_samples, n_features).

    Returns:
        tuple: (NearestNeighbors, build_time_seconds)
    """
    start_time = time.time()
    nn = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(tfidf_matrix)
    return nn, time.time() - start_time

def elastic_index(es, index_name, code_snippets, tfidf_matrix):
    """
    Index code snippets and their vectors in Elasticsearch.

    Args:
        es (Elasticsearch): Authenticated Elasticsearch client.
        index_name (str): Index name.
        code_snippets (list): List of code snippet strings.
        tfidf_matrix (np.ndarray): Array with shape (n_samples, n_features).

    Returns:
        float: Indexing time in seconds.
    """
    # Define the index mapping for dense_vector field
    mapping = {
        "mappings": {
            "properties": {
                "snippet": {"type": "text"},
                "vector": {"type": "dense_vector", "dims": tfidf_matrix.shape[1]}
            }
        }
    }
    # Create the index with the specified mapping (ignore if already exists)
    es.indices.create(index=index_name, body=mapping, ignore=400)

    # Index the documents
    start_time = time.time()
    for i, snippet in enumerate(code_snippets):
        es.index(index=index_name, id=i, body={"snippet": snippet, "vector": tfidf_matrix[i].tolist()})
    return time.time() - start_time

def measure_indexing_times(code_snippets, vectorizer, num_components_list, es):
    """
    Measure indexing times for different methods and dataset sizes.

    Args:
        code_snippets (list): All code snippet strings.
        vectorizer (TfidfVectorizer): Fitted vectorizer.
        num_components_list (list): List of dataset sizes to test.
        es (Elasticsearch): Elasticsearch client.

    Returns:
        dict: Mapping method name to list of times (seconds).
    """
    times = {"faiss": [], "annoy": [], "hnsw": [], "sklearn": [], "elastic": []}
    print("Measuring indexing times...")

    for num_components in num_components_list:
        if num_components > len(code_snippets):
            # If required, repeat data to get larger datasets
            factor = num_components / len(code_snippets)
            tfidf_matrix = np.tile(vectorizer.transform(code_snippets).toarray(), (int(factor), 1))
            code_snippets_large = code_snippets * int(factor)
        else:
            tfidf_matrix = vectorizer.transform(code_snippets[:num_components]).toarray()
            code_snippets_large = code_snippets[:num_components]

        print(f"Indexing {num_components} components...")

        try:
            _, faiss_time = faiss_index(tfidf_matrix)
            times["faiss"].append(faiss_time)
        except MemoryError:
            times["faiss"].append(None)

        try:
            _, annoy_time = annoy_index(tfidf_matrix, tfidf_matrix.shape[1])
            times["annoy"].append(annoy_time)
        except Exception as e:
            times["annoy"].append(None)
            print(f"Error in Annoy indexing: {e}")

        try:
            _, hnsw_time = hnsw_index(tfidf_matrix)
            times["hnsw"].append(hnsw_time)
        except Exception as e:
            times["hnsw"].append(None)
            print(f"Error in HNSW indexing: {e}")

        try:
            _, sklearn_time = sklearn_index(tfidf_matrix)
            times["sklearn"].append(sklearn_time)
        except Exception as e:
            times["sklearn"].append(None)
            print(f"Error in sklearn indexing: {e}")

        try:
            elastic_time = elastic_index(es, "code_snippets", code_snippets_large, tfidf_matrix)
            times["elastic"].append(elastic_time)
        except Exception as e:
            times["elastic"].append(None)
            print(f"Error in Elastic indexing: {e}")

    return times

def main():
    """
    Main entry point:
    - Loads code snippets.
    - Vectorizes them.
    - Measures indexing times for multiple dataset sizes and backends.
    - Prints results.
    """
    filename = "data\data.jsonl"
    print(f"Reading code snippets from {filename}...")
    code_snippets = read_code_snippets(filename)
    
    if not code_snippets:
        print("No code snippets found. Exiting.")
        return

    print(f"Read {len(code_snippets)} code snippets.")
    tfidf_matrix, vectorizer = vectorize_snippets(code_snippets)

    num_components_list = [100, 1000, 10000, 100000]

    es = Elasticsearch(["http://localhost:9200"], basic_auth=('elastic', 'nGRuZaLMB0zgEHHLFLzz'))
    times = measure_indexing_times(code_snippets, vectorizer, num_components_list, es)

    for method, time_list in times.items():
        for num_components, time_taken in zip(num_components_list, time_list):
            if time_taken is not None:
                print(f"{method.upper()}: Indexing time for {num_components} components: {time_taken:.4f} sec")
            else:
                print(f"{method.upper()}: Indexing time for {num_components} components: Error or MemoryError")

if __name__ == "__main__":
    main()
