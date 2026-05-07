"""Measure indexing-time scalability across similarity-search backends."""

import argparse
import os
import time

import faiss
import hnswlib
import numpy as np
from annoy import AnnoyIndex
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from src.similarity_search.utils import read_code_snippets


def vectorize_snippets(code_snippets):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(code_snippets).toarray().astype(np.float32)
    return tfidf_matrix, vectorizer


def faiss_index(tfidf_matrix):
    t0 = time.perf_counter()
    index = faiss.IndexFlatL2(tfidf_matrix.shape[1])
    index.add(tfidf_matrix)
    return index, time.perf_counter() - t0


def annoy_index(tfidf_matrix, dimension):
    t0 = time.perf_counter()
    index = AnnoyIndex(dimension, "angular")
    for i, vector in enumerate(tfidf_matrix):
        index.add_item(i, vector)
    index.build(10)
    return index, time.perf_counter() - t0


def hnsw_index(tfidf_matrix, space="l2"):
    t0 = time.perf_counter()
    index = hnswlib.Index(space=space, dim=tfidf_matrix.shape[1])
    index.init_index(max_elements=len(tfidf_matrix), ef_construction=200, M=16)
    index.add_items(tfidf_matrix)
    return index, time.perf_counter() - t0


def sklearn_index(tfidf_matrix, k):
    t0 = time.perf_counter()
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(tfidf_matrix)
    return nn, time.perf_counter() - t0


def elastic_index(es, index_name, code_snippets, tfidf_matrix):
    mapping = {
        "mappings": {
            "properties": {
                "snippet": {"type": "text"},
                "vector": {"type": "dense_vector", "dims": tfidf_matrix.shape[1]},
            }
        }
    }
    es.indices.create(index=index_name, body=mapping, ignore=400)

    t0 = time.perf_counter()
    for i, snippet in enumerate(code_snippets):
        es.index(index=index_name, id=i, body={"snippet": snippet, "vector": tfidf_matrix[i].tolist()})
    return time.perf_counter() - t0


def measure_indexing_times(code_snippets, vectorizer, num_components_list, es, k, es_index):
    times = {"faiss": [], "annoy": [], "hnsw": [], "sklearn": [], "elastic": []}

    for num_components in num_components_list:
        if num_components > len(code_snippets):
            repeats = int(np.ceil(num_components / len(code_snippets)))
            sampled_snippets = (code_snippets * repeats)[:num_components]
        else:
            sampled_snippets = code_snippets[:num_components]

        tfidf_matrix = vectorizer.transform(sampled_snippets).toarray().astype(np.float32)
        print(f"Indexing {num_components} components...")

        for method in times:
            times[method].append(None)

        try:
            _, times["faiss"][-1] = faiss_index(tfidf_matrix)
        except Exception as exc:
            print(f"Error in FAISS indexing: {exc}")
        try:
            _, times["annoy"][-1] = annoy_index(tfidf_matrix, tfidf_matrix.shape[1])
        except Exception as exc:
            print(f"Error in Annoy indexing: {exc}")
        try:
            _, times["hnsw"][-1] = hnsw_index(tfidf_matrix)
        except Exception as exc:
            print(f"Error in HNSW indexing: {exc}")
        try:
            _, times["sklearn"][-1] = sklearn_index(tfidf_matrix, k)
        except Exception as exc:
            print(f"Error in sklearn indexing: {exc}")
        try:
            times["elastic"][-1] = elastic_index(es, es_index, sampled_snippets, tfidf_matrix)
        except Exception as exc:
            print(f"Error in Elastic indexing: {exc}")

    return times


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.path.join("data", "data.jsonl"))
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 1000, 10000, 100000])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--es-index", default="code_snippets")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    code_snippets = read_code_snippets(args.data)

    if not code_snippets:
        print("No code snippets found. Exiting.")
        return

    _, vectorizer = vectorize_snippets(code_snippets)
    es = Elasticsearch(
        [os.environ.get("ES_HOST", "http://localhost:9200")],
        basic_auth=(
            os.environ.get("ES_USER", "elastic"),
            os.environ.get("ES_PASSWORD", ""),
        ),
    )

    times = measure_indexing_times(code_snippets, vectorizer, args.sizes, es, args.k, args.es_index)

    for method, time_list in times.items():
        for num_components, time_taken in zip(args.sizes, time_list):
            if time_taken is not None:
                print(f"{method.upper()}: Indexing time for {num_components} components: {time_taken:.4f} sec")
            else:
                print(f"{method.upper()}: Indexing time for {num_components} components: Error")


if __name__ == "__main__":
    main()
