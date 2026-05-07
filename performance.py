"""Benchmark latency and throughput for similarity-search backends."""

import argparse
import os

import faiss
import hnswlib
import numpy as np
from annoy import AnnoyIndex
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from src.similarity_search.utils import benchmark_search, read_code_snippets


def vectorize_snippets(code_snippets):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(code_snippets).toarray().astype(np.float32)
    return tfidf_matrix, vectorizer


def faiss_search(tfidf_matrix, query_vector, k):
    index = faiss.IndexFlatL2(tfidf_matrix.shape[1])
    index.add(tfidf_matrix)
    _, indices = index.search(query_vector, k)
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


def sklearn_search(tfidf_matrix, query_vector, k):
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(tfidf_matrix)
    _, indices = nn.kneighbors(query_vector)
    return indices


def elastic_search(es, index_name, query_vector, k):
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                "params": {"query_vector": query_vector.tolist()},
            },
        }
    }
    response = es.search(index=index_name, body={"query": script_query, "size": k})
    return [hit["_source"]["snippet"] for hit in response["hits"]["hits"]]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.path.join("data", "data.jsonl"))
    parser.add_argument("--query", default="int result = 1; for (int i = 1; i <= n; i++) { result *= i; }")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--es-index", default="code_snippets_index99")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    code_snippets = read_code_snippets(args.data)
    tfidf_matrix, vectorizer = vectorize_snippets(code_snippets)
    query_vector = vectorizer.transform([args.query]).toarray().astype(np.float32)

    es = Elasticsearch(
        [os.environ.get("ES_HOST", "http://localhost:9200")],
        basic_auth=(
            os.environ.get("ES_USER", "elastic"),
            os.environ.get("ES_PASSWORD", ""),
        ),
    )

    faiss_time, faiss_qps = benchmark_search(faiss_search, (tfidf_matrix, query_vector, args.k), args.trials, "FAISS")
    annoy_time, annoy_qps = benchmark_search(
        annoy_search, (tfidf_matrix, query_vector, args.k, tfidf_matrix.shape[1]), args.trials, "Annoy"
    )
    hnsw_time, hnsw_qps = benchmark_search(hnsw_search, (tfidf_matrix, query_vector, args.k), args.trials, "HNSWlib")
    sklearn_time, sklearn_qps = benchmark_search(
        sklearn_search, (tfidf_matrix, query_vector, args.k), args.trials, "Scikit-learn"
    )

    svd = TruncatedSVD(n_components=min(4096, tfidf_matrix.shape[1] - 1))
    svd.fit(tfidf_matrix)
    normalized_query_vector = normalize(svd.transform(vectorizer.transform([args.query]).toarray()), norm="l2")
    elastic_time, elastic_qps = benchmark_search(
        elastic_search,
        (es, args.es_index, normalized_query_vector[0], args.k),
        args.trials,
        "Elasticsearch",
    )

    print(f"FAISS: {faiss_time:.4f} sec/query, {faiss_qps:.2f} queries/sec")
    print(f"Annoy: {annoy_time:.4f} sec/query, {annoy_qps:.2f} queries/sec")
    print(f"HNSWlib: {hnsw_time:.4f} sec/query, {hnsw_qps:.2f} queries/sec")
    print(f"Scikit-learn: {sklearn_time:.4f} sec/query, {sklearn_qps:.2f} queries/sec")
    print(f"Elastic: {elastic_time:.4f} sec/query, {elastic_qps:.2f} queries/sec")


if __name__ == "__main__":
    main()
