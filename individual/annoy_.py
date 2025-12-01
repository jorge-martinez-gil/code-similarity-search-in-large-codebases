import json
from sklearn.feature_extraction.text import TfidfVectorizer
from annoy import AnnoyIndex

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

filename = "data.jsonl"  # Replace with your actual file name
code_snippets = read_code_snippets(filename)

# Vectorize the code snippets
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(code_snippets).toarray()

# Initialize Annoy index
dimension = tfidf_matrix.shape[1]  # Dimension of the vectors
index = AnnoyIndex(dimension, 'angular')  # Using Angular distance

# Add items to the index
for i, vector in enumerate(tfidf_matrix):
    index.add_item(i, vector)

# Build the index
index.build(10)  # The number of trees for building the index

# Vectorize the query code snippet
query_snippet = "int result = 1; for (int i = 1; i <= n; i++) { result *= i; }"
query_vector = vectorizer.transform([query_snippet]).toarray()[0]

# Perform the similarity search
k = 3  # Number of nearest neighbors to find
nearest_neighbors = index.get_nns_by_vector(query_vector, k, include_distances=True)

# Print the most similar code snippets
print("Most similar code snippets to your query:")
for i in nearest_neighbors[0]:
    print(code_snippets[i])
