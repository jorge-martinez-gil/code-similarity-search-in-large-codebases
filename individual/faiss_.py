import faiss
import numpy as np
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

# Step 1: Preprocess code snippets
filename = "data.jsonl"  
code_snippets = read_code_snippets(filename)

# Step 2: Vectorize the code snippets
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(code_snippets).toarray()

# Step 3: Index the vectors using FAISS
dimension = tfidf_matrix.shape[1]  # Dimension of the vectors
index = faiss.IndexFlatL2(dimension)  # Use the L2 distance for similarity
index.add(tfidf_matrix.astype(np.float32))  # FAISS expects vectors to be in float32

# Step 4: Vectorize the query code snippet
query_snippet = "int result = 1; for (int i = 1; i <= n; i++) { result *= i; }"
query_vector = vectorizer.transform([query_snippet]).toarray().astype(np.float32)

# Step 5: Perform the similarity search
k = 3  # Number of nearest neighbors to find
D, I = index.search(query_vector, k)  # D is the distance, I is the index of neighbors

# Print the most similar code snippets
print("Most similar code snippets to your query:")
for i in I[0]:
    print(code_snippets[i])
