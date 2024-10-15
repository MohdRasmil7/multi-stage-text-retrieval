from sentence_transformers import SentenceTransformer
import numpy as np
from dataset_preparation import load_dataset

def embed_text(model, texts):
    return model.encode(texts, convert_to_tensor=True)

def retrieve_top_k(query_embedding, corpus_embeddings, k=10):
    similarities = np.dot(query_embedding, corpus_embeddings.T)
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    return top_k_indices

if __name__ == "__main__":
    # Load the dataset
    corpus, queries, _ = load_dataset("path_to_your_dataset")

    # Load the embedding models
    small_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    corpus_embeddings = embed_text(small_model, list(corpus.values()))
    
    # Embed the query
    query = ["What is the impact of climate change on polar bears?"]
    query_embedding = embed_text(small_model, query)

    # Retrieve the top-10 documents
    top_k_indices = retrieve_top_k(query_embedding, corpus_embeddings, k=10)
    top_k_passages = [list(corpus.keys())[i] for i in top_k_indices]
    
    print("Top-10 Retrieved Passages:")
    print(top_k_passages)
