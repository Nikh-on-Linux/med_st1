import ray
import json
import logging
import sys
import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchRequest, Filter, FieldCondition, MatchValue

# --- Configuration ---

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Qdrant & Model Configuration ---
QDRANT_HOST = '172.25.120.212'
QDRANT_PORT = 6333
COLLECTION_NAME = 'med_st1'

# 1. Retriever Model (Bi-Encoder) - The "Key"
RETRIEVER_MODEL_NAME = 'ncbi/MedCPT-Query-Encoder'
# 2. Re-ranker Model (Cross-Encoder) - The "Judge"
RERANKER_MODEL_NAME = 'ncbi/MedCPT-Cross-Encoder' 

TOP_K_RETRIEVAL = 100  # How many results to fetch from Qdrant (the "broad" search)
TOP_K_FINAL = 5       # How many results to show the user after re-ranking

# --- Embedding Function ---

def _get_cls_embedding(model_output, attention_mask):
    """
    MedCPT uses the [CLS] token embedding.
    This is a helper to extract it.
    """
    cls_embedding = model_output.last_hidden_state[:, 0]
    return cls_embedding

def get_query_embedding(query, model, tokenizer, device):
    """
    Generates a single vector embedding for a text query using the Retriever.
    """
    logging.info("Generating query embedding...")
    with torch.no_grad():
        inputs = tokenizer(
            [query],  # Note: needs to be a list
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        ).to(device)
        
        outputs = model(**inputs)
        embedding_tensor = _get_cls_embedding(outputs, inputs['attention_mask'])
        vector = embedding_tensor.cpu().numpy()[0].tolist()
        
    logging.info("Query embedding generated.")
    return vector

# --- Qdrant Search Function ---

def search_collection(client, collection_name, query_vector, top_k):
    """
    Performs the initial semantic search on the Qdrant collection.
    """
    logging.info(f"Retrieving top {top_k} candidates from collection '{collection_name}'...")
    try:
        # FIX: Use query_points instead of deprecated search
        search_results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )
        return search_results.points  # query_points returns a QueryResponse object
        
    except Exception as e:
        logging.error(f"Failed to search Qdrant collection: {e}")
        return None

# --- Re-ranking Function ---

def rerank_results(query, results, model, tokenizer, device):
    """
    Uses the Cross-Encoder model to re-score the retrieved results.
    Returns a list of tuples: (result, rerank_score)
    """
    logging.info(f"Re-ranking {len(results)} candidates...")
    
    # Create pairs of [query, summary]
    pairs = []
    for hit in results:
        summary = hit.payload.get('summary', '')
        pairs.append([query, summary])
        
    if not pairs:
        return []

    # Tokenize and run the re-ranker model in a batch
    with torch.no_grad():
        inputs = tokenizer(
            pairs, 
            padding=True, 
            truncation=True, 
            return_tensors='pt', 
            max_length=512
        ).to(device)
        
        outputs = model(**inputs)
        
        # The output of MedCPT-Cross-Encoder is a single logit per pair.
        # Higher is better.
        scores = outputs.logits.squeeze().cpu().numpy()
    
    # FIX: Create a new list of tuples instead of modifying Pydantic objects
    reranked_results = []
    for i, hit in enumerate(results):
        # Store as tuple: (original_hit, rerank_score)
        reranked_results.append((hit, float(scores[i])))
        
    # Sort by rerank_score in descending order
    reranked_results.sort(key=lambda x: x[1], reverse=True)
    
    logging.info("Re-ranking complete.")
    return reranked_results

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Query and re-rank medical abstracts from Qdrant.")
    parser.add_argument(
        "query",
        type=str,
        help="The search query (e.g., 'what is the link between heart disease and diabetes?')."
    )
    args = parser.parse_args()
    
    query_text = args.query
    logging.info(f"Received query: \"{query_text}\"")

    # --- 1. Load Models ---
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Loading models on device: {device}")
        
        # Load Retriever (Query Encoder)
        retriever_tokenizer = AutoTokenizer.from_pretrained(RETRIEVER_MODEL_NAME)
        retriever_model = AutoModel.from_pretrained(RETRIEVER_MODEL_NAME).to(device)
        retriever_model.eval()
        
        # Load Re-ranker (Cross-Encoder)
        reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
        reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME).to(device)
        reranker_model.eval()
        
        logging.info("All models loaded successfully.")
    except Exception as e:
        logging.critical(f"Failed to load models. Error: {e}")
        logging.critical("Please ensure 'transformers' and 'torch' are installed (`pip install -r requirements.txt`)")
        sys.exit(1)

    # --- 2. Generate Query Embedding ---
    query_vector = get_query_embedding(
        query_text, 
        retriever_model, 
        retriever_tokenizer, 
        device
    )

    # --- 3. Connect to Qdrant and Retrieve Initial Candidates ---
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        client.get_collections() # Health check
        logging.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}.")
    except Exception as e:
        logging.critical(f"Failed to connect to Qdrant. Is it running at {QDRANT_HOST}:{QDRANT_PORT}?")
        logging.critical(f"Error: {e}")
        sys.exit(1)
        
    initial_results = search_collection(
        client, 
        COLLECTION_NAME, 
        query_vector, 
        TOP_K_RETRIEVAL
    )

    if not initial_results:
        logging.warning("No initial results found from Qdrant.")
        return

    # --- 4. Re-rank the Candidates ---
    # Returns list of tuples: [(hit, rerank_score), ...]
    final_results = rerank_results(
        query_text, 
        initial_results, 
        reranker_model, 
        reranker_tokenizer, 
        device
    )

    # --- 5. Display Final Results ---
    if final_results:
        # Get the top K final results
        top_final_results = final_results[:TOP_K_FINAL]
        
        print(f"\n--- Top {len(top_final_results)} Re-ranked Results for: \"{query_text}\" ---\n")
        for i, (hit, rerank_score) in enumerate(top_final_results):
            print(f"Result {i+1} (Re-rank Score: {rerank_score:.4f} | Initial Score: {hit.score:.4f}):")
            print(f"  Doc ID:  {hit.payload.get('doc_id', 'N/A')}")
            print(f"  Labels:  {hit.payload.get('labels', 'N/A')}")
            print(f"  Summary: {hit.payload.get('summary', 'N/A')}")
            print("-" * 20)
    else:
        logging.warning("No results found after re-ranking.")
        
if __name__ == "__main__":
    main()