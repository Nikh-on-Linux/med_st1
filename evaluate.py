import json
import logging
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from qdrant_client import QdrantClient
from tqdm import tqdm

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Qdrant & Model Configuration ---
QDRANT_HOST = '172.25.120.212'
QDRANT_PORT = 6333
COLLECTION_NAME = 'med_st1'
RETRIEVER_MODEL_NAME = 'ncbi/MedCPT-Query-Encoder'
RERANKER_MODEL_NAME = 'ncbi/MedCPT-Cross-Encoder'

# --- Evaluation Parameters ---
EVAL_TOP_K = 10         # We will check metrics (Recall, MRR) up to this rank
RERANKER_FETCH_K = 100   # How many docs to pull from Qdrant to give to the re-ranker
SAMPLE_SIZE = 100       # Set to 'None' to run all 1000, or a number for a quick test

# --- Helper Functions (from search_qdrant.py) ---

def _get_cls_embedding(model_output, attention_mask):
    cls_embedding = model_output.last_hidden_state[:, 0]
    return cls_embedding

def get_query_embedding(query, model, tokenizer, device):
    with torch.no_grad():
        inputs = tokenizer(
            [query], padding=True, truncation=True, return_tensors='pt', max_length=512
        ).to(device)
        outputs = model(**inputs)
        embedding_tensor = _get_cls_embedding(outputs, inputs['attention_mask'])
        return embedding_tensor.cpu().numpy()[0].tolist()

def search_qdrant(client, vector, k):
    try:
        search_results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=k,
            with_payload=True
        )
        return search_results.points
    except Exception as e:
        logging.error(f"Qdrant search failed: {e}")
        return []

def rerank_results(query, results, model, tokenizer, device):
    pairs = []
    for hit in results:
        summary = hit.payload.get('summary', '')
        pairs.append([query, summary])
        
    if not pairs:
        return []

    with torch.no_grad():
        inputs = tokenizer(
            pairs, padding=True, truncation=True, return_tensors='pt', max_length=512
        ).to(device)
        outputs = model(**inputs)
        scores = outputs.logits.squeeze().cpu().numpy()
    
    reranked_list = []
    for i, hit in enumerate(results):
        reranked_list.append({
            'original_hit': hit,
            'rerank_score': scores[i]
        })
        
    reranked_list.sort(key=lambda x: x['rerank_score'], reverse=True)
    return reranked_list

# --- Evaluation Metrics Calculator ---

class RetrievalMetrics:
    def __init__(self, k):
        self.k = k
        self.hits_at_k = {i: 0 for i in range(1, k + 1)}
        self.reciprocal_ranks = []
        self.total_queries = 0

    def add_result(self, ranked_doc_ids, expected_doc_id):
        self.total_queries += 1
        rank = 0
        found = False
        
        for i, doc_id in enumerate(ranked_doc_ids):
            if doc_id == expected_doc_id:
                rank = i + 1
                found = True
                break
        
        if found:
            for i in range(rank, self.k + 1):
                self.hits_at_k[i] += 1
            self.reciprocal_ranks.append(1 / rank)
        else:
            self.reciprocal_ranks.append(0)

    def calculate_metrics(self):
        if self.total_queries == 0:
            return {}
            
        recall_at_k = {
            f"R@{i}": self.hits_at_k[i] / self.total_queries
            for i in self.hits_at_k
        }
        
        mrr_at_k = {
            f"MRR@{self.k}": np.mean(self.reciprocal_ranks)
        }
        
        return {**recall_at_k, **mrr_at_k}

# --- Main Evaluation Script ---

def main():
    # --- 1. Load Ground Truth Data ---
    logging.info("Loading ground truth data...")
    try:
        # Load summaries to know which doc_ids are in Qdrant
        with open('llm_responses.json', 'r') as f:
            llm_data = json.load(f)
        # Load raw data to get the corresponding questions
        with open('./data/raw.json', 'r') as f:
            raw_data = json.load(f)
    except FileNotFoundError as e:
        logging.critical(f"Error loading file: {e}. Make sure 'llm_responses.json' and 'raw.json' are in the same directory.")
        return

    # Create the test set: (query, expected_doc_id)
    test_set = []
    for doc_id, data in llm_data.items():
        if "PARSE_ERROR" not in data['llm_response'] and "UNHANDLED_WORKER_ERROR" not in data['llm_response']:
            if doc_id in raw_data and "QUESTION" in raw_data[doc_id]:
                test_set.append({
                    "query": raw_data[doc_id]["QUESTION"],
                    "expected_id": doc_id
                })
    
    if SAMPLE_SIZE and SAMPLE_SIZE < len(test_set):
        logging.info(f"Using a random sample of {SAMPLE_SIZE} queries for evaluation.")
        indices = np.random.choice(len(test_set), SAMPLE_SIZE, replace=False)
        test_set = [test_set[i] for i in indices]
    else:
        logging.info(f"Evaluating on all {len(test_set)} queries.")

    # --- 2. Load Models and Connect to Qdrant ---
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Loading models on device: {device}")
        
        retriever_tokenizer = AutoTokenizer.from_pretrained(RETRIEVER_MODEL_NAME)
        retriever_model = AutoModel.from_pretrained(RETRIEVER_MODEL_NAME).to(device)
        retriever_model.eval()
        
        reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
        reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME).to(device)
        reranker_model.eval()
        
        logging.info("All models loaded successfully.")
    except Exception as e:
        logging.critical(f"Failed to load models: {e}")
        return

    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        client.get_collections() # Health check
        logging.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}.")
    except Exception as e:
        logging.critical(f"Failed to connect to Qdrant: {e}")
        return

    # --- 3. Run Evaluation ---
    metrics_retriever_only = RetrievalMetrics(k=EVAL_TOP_K)
    metrics_with_reranker = RetrievalMetrics(k=EVAL_TOP_K)

    logging.info("Starting evaluation...")
    for item in tqdm(test_set, desc="Evaluating Queries"):
        query = item["query"]
        expected_id = item["expected_id"]
        
        # Generate the query embedding
        query_vector = get_query_embedding(query, retriever_model, retriever_tokenizer, device)

        # --- Test 1: Retriever Only ---
        retriever_results = search_qdrant(client, query_vector, k=EVAL_TOP_K)
        retriever_doc_ids = [hit.payload.get("doc_id") for hit in retriever_results]
        metrics_retriever_only.add_result(retriever_doc_ids, expected_id)

        # --- Test 2: Retriever + Re-ranker ---
        # 1. Fetch more candidates
        reranker_candidates = search_qdrant(client, query_vector, k=RERANKER_FETCH_K)
        # 2. Re-rank them
        reranked_list = rerank_results(query, reranker_candidates, reranker_model, reranker_tokenizer, device)
        # 3. Get the top K doc_ids after re-ranking
        reranked_doc_ids = [item['original_hit'].payload.get("doc_id") for item in reranked_list[:EVAL_TOP_K]]
        metrics_with_reranker.add_result(reranked_doc_ids, expected_id)

    # --- 4. Print Final Report ---
    logging.info("Evaluation complete. Calculating metrics...")
    
    final_metrics_retriever = metrics_retriever_only.calculate_metrics()
    final_metrics_reranker = metrics_with_reranker.calculate_metrics()

    print("\n" + "="*30)
    print("  Retrieval Evaluation Report")
    print("="*30)
    print(f"\nEvaluated on {len(test_set)} queries.\n")

    print("--- Baseline (Retriever Only) ---")
    print(f"  MRR@{EVAL_TOP_K}:   {final_metrics_retriever.get(f'MRR@{EVAL_TOP_K}', 0):.4f}")
    print(f"  Recall@1:  {final_metrics_retriever.get('R@1', 0) * 100:.2f}%")
    print(f"  Recall@5:  {final_metrics_retriever.get('R@5', 0) * 100:.2f}%")
    print(f"  Recall@10: {final_metrics_retriever.get(f'R@{EVAL_TOP_K}', 0) * 100:.2f}%")

    print("\n--- Pipeline (Retriever + Re-ranker) ---")
    print(f"  MRR@{EVAL_TOP_K}:   {final_metrics_reranker.get(f'MRR@{EVAL_TOP_K}', 0):.4f}")
    print(f"  Recall@1:  {final_metrics_reranker.get('R@1', 0) * 100:.2f}%")
    print(f"  Recall@5:  {final_metrics_reranker.get('R@5', 0) * 100:.2f}%")
    print(f"  Recall@10: {final_metrics_reranker.get(f'R@{EVAL_TOP_K}', 0) * 100:.2f}%")
    print("\n" + "="*30)

if __name__ == "__main__":
    main()