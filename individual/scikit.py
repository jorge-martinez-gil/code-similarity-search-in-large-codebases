import json
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
import numpy as np
from tqdm import tqdm

def read_code_snippets(filename):
    code_snippets = []
    with open(filename, 'r') as file:
        for line in file:
            data = json.loads(line)
            code_snippet = data["func"]
            code_snippets.append(code_snippet)
    return code_snippets

def vectorize_snippets(code_snippets):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(code_snippets).toarray()
    return tfidf_matrix, vectorizer

def sklearn_search(tfidf_matrix, query_vector, k):
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(tfidf_matrix)
    distances, indices = nn.kneighbors(query_vector)
    return indices

def benchmark_search(function, args, num_trials=5):
    start_time = time.time()
    for _ in tqdm(range(num_trials), desc=f"Benchmarking {function.__name__}"):
        results = function(*args)
    elapsed_time = time.time() - start_time
    throughput = num_trials / elapsed_time
    return elapsed_time / num_trials, throughput

def main():
    filename = "data.jsonl"
    query_snippet = "int result = 1; for (int i = 1; i <= n; i++) { result *= i; }"
    code_snippets = read_code_snippets(filename)
    tfidf_matrix, vectorizer = vectorize_snippets(code_snippets)
    query_vector = vectorizer.transform([query_snippet]).toarray()

    # Benchmark Scikit-learn NearestNeighbors
    sklearn_time, sklearn_throughput = benchmark_search(sklearn_search, (tfidf_matrix, query_vector, 3))

    print(f"Scikit-learn NearestNeighbors: {sklearn_time:.4f} sec/query, {sklearn_throughput:.2f} queries/sec")

if __name__ == "__main__":
    main()
