import json
from sklearn.feature_extraction.text import TfidfVectorizer
import hnswlib
import numpy as np

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

def hnsw_search(tfidf_matrix, query_vector, k, space='cosine', ef=200, M=16):
    dimension = tfidf_matrix.shape[1]
    p = hnswlib.Index(space=space, dim=dimension)
    p.init_index(max_elements=len(tfidf_matrix), ef_construction=ef, M=M)
    p.add_items(tfidf_matrix)
    
    # Controlling the search speed/accuracy trade-off
    p.set_ef(ef)  # ef should always be > k
    
    # Query for the nearest neighbors of the vector
    labels, distances = p.knn_query(query_vector, k=k)
    return labels, distances


filename = "data.jsonl"
query_snippet = "int result = 1; for (int i = 1; i <= n; i++) { result *= i; }"
code_snippets = read_code_snippets(filename)
tfidf_matrix, vectorizer = vectorize_snippets(code_snippets)
query_vector = vectorizer.transform([query_snippet]).toarray()

# Assuming nearest_neighbors is the result of hnsw_search
# and you have requested k nearest neighbors:

# Perform the search
k = 3  # Number of nearest neighbors to find
nearest_neighbors, distances = hnsw_search(tfidf_matrix, query_vector, k)

# Print the most similar code snippets
print("Most similar code snippets to your query:")
for idx in nearest_neighbors[0]:  # Assuming a single query vector was used
    print(code_snippets[idx])

