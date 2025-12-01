!pip install faiss-cpu
!pip install scann
!pip install annoy
!pip install tikzplotlib
!pip install hnswlib


import json
import time
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import faiss
from annoy import AnnoyIndex
import hnswlib
from transformers import AutoModel, AutoTokenizer

def read_code_snippets(filename):
    code_snippets = []
    with open(filename, 'r') as file:
        for line in file:
            data = json.loads(line)
            code_snippet = data["func"]
            code_snippets.append(code_snippet)
    return code_snippets

def vectorize_snippets_tfidf(code_snippets):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(code_snippets).toarray()
    return tfidf_matrix, vectorizer

def vectorize_snippets_codebert(code_snippets):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    embeddings = []
    for snippet in code_snippets:
        inputs = tokenizer(snippet, return_tensors='pt', truncation=True, padding=True)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy()[0])
    return np.array(embeddings)

def benchmark_search(function, args, num_trials=5):
    start_time = time.time()
    for _ in tqdm(range(num_trials), desc='Processing searches'):
        function(*args)
    elapsed_time = time.time() - start_time
    throughput = num_trials / elapsed_time
    return elapsed_time / num_trials, throughput

def faiss_search(tfidf_matrix, query_vector, k):
    dimension = tfidf_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(tfidf_matrix.astype(np.float32))
    _, I = index.search(query_vector.astype(np.float32), k)
    return I

def annoy_search(tfidf_matrix, query_vector, k, dimension):
    index = AnnoyIndex(dimension, 'angular')
    for i, vector in enumerate(tfidf_matrix):
        index.add_item(i, vector)
    index.build(10)
    I = index.get_nns_by_vector(query_vector[0], k, include_distances=False)
    return I

def hnsw_search(tfidf_matrix, query_vector, k, space='l2', dimension=None):
    dimension = dimension or tfidf_matrix.shape[1]
    p = hnswlib.Index(space=space, dim=dimension)
    p.init_index(max_elements=len(tfidf_matrix), ef_construction=200, M=16)
    p.add_items(tfidf_matrix)
    p.set_ef(200)
    labels, distances = p.knn_query(query_vector, k)
    return labels

def create_es_index(es, index_name, dimension):
    mappings = {
        "mappings": {
            "properties": {
                "snippet": {"type": "text"},
                "vector": {"type": "dense_vector", "dims": dimension}
            }
        }
    }
    es.indices.create(index=index_name, body=mappings)

def index_code_snippets(es, index_name, code_snippets, tfidf_matrix):
    start_time = time.time()
    for i, snippet in enumerate(code_snippets):
        es.index(index=index_name, id=i, body={"snippet": snippet, "vector": tfidf_matrix[i].tolist()})
    indexing_time = time.time() - start_time
    return indexing_time

def elastic_search(es, index_name, query_vector, k):
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                "params": {"query_vector": query_vector.tolist()}
            }
        }
    }
    try:
        response = es.search(index=index_name, body={"query": script_query, "size": k})
    except Exception as e:
        print(f"Elasticsearch query failed: {e}")
        return []
    return [hit["_source"]["snippet"] for hit in response["hits"]["hits"]]

def sklearn_nn_search(tfidf_matrix, query_vector, k):
    nn = NearestNeighbors(n_neighbors=k, metric='cosine')
    nn.fit(tfidf_matrix)
    distances, indices = nn.kneighbors(query_vector)
    return indices

def evaluate_accuracy(search_function, search_args, ground_truth, k=3):
    retrieved_docs = search_function(*search_args)
    correct = sum([1 for doc in retrieved_docs if doc in ground_truth])
    return correct / k

def main():
    filename = "data\data.jsonl"
    query_snippet = "int result = 1; for (int i = 1; i <= n; i++) { result *= i; }"
    code_snippets = read_code_snippets(filename)

    tfidf_matrix, tfidf_vectorizer = vectorize_snippets_tfidf(code_snippets)
    query_vector_tfidf = tfidf_vectorizer.transform([query_snippet]).toarray()

    codebert_matrix = vectorize_snippets_codebert(code_snippets)
    query_vector_codebert = vectorize_snippets_codebert([query_snippet])

    # Benchmark each method
    methods = [
        ('FAISS', faiss_search, (tfidf_matrix, query_vector_tfidf, 3)),
        ('SklearnNN', sklearn_nn_search, (tfidf_matrix, query_vector_tfidf, 3)),
        ('Annoy', annoy_search, (tfidf_matrix, query_vector_tfidf, 3, dimension)),
        ('HNSW', hnsw_search, (tfidf_matrix, query_vector_tfidf, 3, 'l2', dimension))
    ]

    times = []
    throughputs = []

    for name, function, args in methods:
        avg_time, throughput = benchmark_search(function, args, num_trials=10000)
        times.append(avg_time)
        throughputs.append(throughput)
        print(f"{name}: {avg_time:.4f} sec/query, {throughput:.2f} queries/sec")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.bar([method[0] for method in methods], times, color='blue')
    plt.xlabel('Search Method')
    plt.ylabel('Average Time per Query (s)')
    plt.title('Search Performance Comparison')
    tikzplotlib.save("search_time_comparison.tex")

    plt.figure(figsize=(10, 5))
    plt.bar([method[0] for method in methods], throughputs, color='green')
    plt.xlabel('Search Method')
    plt.ylabel('Queries per Second')
    plt.title('Search Throughput Comparison')
    tikzplotlib.save("search_throughput_comparison.tex")

if __name__ == "__main__":
    main()