import numpy as np
import scann
from sklearn.feature_extraction.text import TfidfVectorizer
import json

def read_code_snippets(filename):
    """
    Reads code snippets from a file with JSON-like formatting.

    Args:
        filename: The path to the file containing the code snippets.

    Returns:
        A list of extracted code snippets.
    """
    code_snippets = []
    with open(filename, 'r') as file:
        for line in file:
            data = json.loads(line)  # Load JSON data from each line
            code_snippet = data["func"]  # Extract the code from the "func" field
            code_snippets.append(code_snippet)

    return code_snippets

# Load code snippets from file
filename = "data.jsonl"
code_snippets = read_code_snippets(filename)

# Vectorize the code snippets
vectorizer = TfidfVectorizer(max_features=500)  # Limit the number of features to reduce vector size
tfidf_matrix = vectorizer.fit_transform(code_snippets).toarray()

# Normalize the vectors (important for cosine similarity)
normalized_matrix = tfidf_matrix / np.linalg.norm(tfidf_matrix, axis=1)[:, np.newaxis]

# Create the ScaNN searcher with minimal configuration
searcher = scann.scann_ops_pybind.builder(normalized_matrix, 10, "dot_product").tree(
    num_leaves=10, num_leaves_to_search=1, training_sample_size=20000
).score_ah(
    dimensions_per_block=1,  # Lowest possible configuration
    anisotropic_quantization_threshold=0.2
).reorder(10).build()  # Minimal reordering pool

# Vectorize the query code snippet
query_snippet = "int result = 1; for (int i = 1; i <= n; i++) { result *= i; }"
query_vector = vectorizer.transform([query_snippet]).toarray()
normalized_query_vector = query_vector / np.linalg.norm(query_vector)

# Perform the similarity search
neighbors, distances = searcher.search(normalized_query_vector[0], final_num_neighbors=3)

# Print the most similar code snippets
print("Most similar code snippets to your query:")
for idx in neighbors:
    print(code_snippets[idx])
