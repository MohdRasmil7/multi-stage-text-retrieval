from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

def rerank_passages(query, passages, model_name):
    # Load ranking model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize input
    inputs = tokenizer([query] * len(passages), passages, padding=True, truncation=True, return_tensors="pt")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits.squeeze().cpu().numpy()

    return np.argsort(scores)[::-1]

if __name__ == "__main__":
    # Sample query and top-k passages from retrieval pipeline
    query = "What is the impact of climate change on polar bears?"
    top_k_passages = ["Passage 1", "Passage 2", "Passage 3"]  # Replace with actual passages
    
    # Rerank the passages
    reranked_indices = rerank_passages(query, top_k_passages, 'cross-encoder/ms-marco-MiniLM-L-12-v2')
    final_ranking = [top_k_passages[i] for i in reranked_indices]
    
    print("Reranked Passages:")
    print(final_ranking)
