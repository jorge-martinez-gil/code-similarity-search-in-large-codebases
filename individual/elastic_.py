import json
from elasticsearch import Elasticsearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
import numpy as np

def read_code_snippets(filename):
    """
    Reads code snippets from a JSONL file.
    """
    code_snippets = []
    with open(filename, 'r') as file:
        for line in file:
            data = json.loads(line)
            code_snippet = data["func"]
            code_snippets.append(code_snippet)
    return code_snippets

def create_es_index(es, index_name, dimension):
    """
    Creates an Elasticsearch index with the specified settings and mappings.
    """
    index_body = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "snippet": {"type": "text"},
                "vector": {
                    "type": "dense_vector",
                    "dims": dimension
                }
            }
        }
    }
    
    try:
        es.indices.create(index=index_name, body=index_body)
        print(f"Index '{index_name}' created successfully.")
    except Exception as e:
        print(f"Failed to create index '{index_name}': {str(e)}")

def index_code_snippets(es, index_name, code_snippets, reduced_tfidf_matrix):
    """
    Indexes code snippets and their corresponding reduced-dimensionality TF-IDF vectors in Elasticsearch.
    """
    for i, snippet in enumerate(code_snippets):
        es.index(index=index_name, id=i, body={"snippet": snippet, "vector": reduced_tfidf_matrix[i].tolist()})

def search_similar_snippets(es, index_name, query_vector, k):
    """
    Searches for similar code snippets based on cosine similarity.
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

# Setup
filename = "data.jsonl"
index_name = "code_snippets_index99"
es = Elasticsearch(
    ["http://localhost:9200"],
    basic_auth=('elastic', 'nGRuZaLMB0zgEHHLFLzz')  # Update with correct credentials
)

# Read and vectorize code snippets
code_snippets = read_code_snippets(filename)
vectorizer = TfidfVectorizer()
tfidf_matrix = normalize(vectorizer.fit_transform(code_snippets).toarray(), norm='l2')

# Dimensionality Reduction
svd = TruncatedSVD(n_components=4096)  # Reduce to 4096 dimensions
reduced_tfidf_matrix = svd.fit_transform(tfidf_matrix)

# Index the vectors using Elasticsearch
if not es.indices.exists(index=index_name):
    dimension = 4096  # Updated dimensionality after reduction
    create_es_index(es, index_name, dimension)
    index_code_snippets(es, index_name, code_snippets, reduced_tfidf_matrix)

# Prepare and search similar snippets
query_snippet = "int result = 1; for (int i = 1; i <= n; i++) { result *= i; }"
query_vector = normalize(svd.transform(vectorizer.transform([query_snippet]).toarray()), norm='l2')
similar_snippets = search_similar_snippets(es, index_name, query_vector[0], k=3)

# Print the most similar code snippets
print("Most similar code snippets to your query:")
for snippet in similar_snippets:
    print(snippet)


