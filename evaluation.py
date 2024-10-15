from sklearn.metrics import ndcg_score

def get_relevance_scores(top_k_passages, qrels, query_id):
    true_relevant_passages = qrels[query_id]
    return [1 if passage in true_relevant_passages else 0 for passage in top_k_passages]

def evaluate_ndcg(top_k_passages, top_k_scores, qrels, query_id):
    relevance_scores = get_relevance_scores(top_k_passages, qrels, query_id)
    ndcg = ndcg_score([relevance_scores], [top_k_scores], k=10)
    return ndcg

if __name__ == "__main__":
    # Example top-k retrieval and relevance scores
    top_k_passages = ["doc1", "doc2", "doc3", "doc4"]
    top_k_scores = [0.9, 0.8, 0.7, 0.6]  # These would be the similarity scores from retrieval
    
    # Assuming qrels contains the relevance judgements for the query
    qrels = {"<query_id>": ["doc1", "doc3"]}  # Replace with actual relevance data
    
    # Evaluate NDCG@10
    ndcg = evaluate_ndcg(top_k_passages, top_k_scores, qrels, "<query_id>")
    print(f"NDCG@10: {ndcg}")
