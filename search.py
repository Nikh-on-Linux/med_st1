import ray
import json
import logging
import sys
import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModel
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
# *** CRITICAL: Use the Query Encoder for search queries ***
MODEL_NAME = 'ncbi/MedCPT-Query-Encoder'
TOP_K = 5  # Number of results to return

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
    Generates a single vector embedding for a text query.
    """
    logging.info("Generating embedding for query...")
    with torch.no_grad():
        # Tokenize the query
        inputs = tokenizer(
            [query],  # Note: needs to be a list
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        ).to(device)
        
        # Get model outputs
        outputs = model(**inputs)
        
        # Get the [CLS] embedding
        embedding_tensor = _get_cls_embedding(outputs, inputs['attention_mask'])
        
        # Move to CPU and convert to a simple list
        vector = embedding_tensor.cpu().numpy()[0].tolist()
        
    logging.info("Embedding generated.")
    return vector

# --- Qdrant Search Function ---

def search_collection(client, collection_name, query_vector, top_k):
    """
    Performs the semantic search on the Qdrant collection.
    """
    logging.info(f"Searching collection '{collection_name}'...")
    try:
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True  # We want to get the metadata back
        )
        return search_results
        
    except Exception as e:
        logging.error(f"Failed to search Qdrant collection: {e}")
        return None

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Query the medical abstracts in Qdrant.")
    parser.add_argument(
        "query",
        type=str,
        help="The search query (e.g., 'what is the link between heart disease and diabetes?')."
    )
    args = parser.parse_args()
    
    query_text = args.query
    logging.info(f"Received query: \"{query_text}\"")

    # --- 1. Load Model ---
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Loading model '{MODEL_NAME}' on device: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME).to(device)
        model.eval()
        
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.critical(f"Failed to load model '{MODEL_NAME}'. Error: {e}")
        logging.critical("Please ensure 'transformers' and 'torch' are installed (`pip install -r requirements.txt`)")
        sys.exit(1)

    # --- 2. Generate Query Embedding ---
    query_vector = get_query_embedding(query_text, model, tokenizer, device)

    # --- 3. Connect to Qdrant and Search ---
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        client.get_collections() # Health check
        logging.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}.")
    except Exception as e:
        logging.critical(f"Failed to connect to Qdrant. Is it running at {QDRANT_HOST}:{QDRANT_PORT}?")
        logging.critical(f"Error: {e}")
        sys.exit(1)
        
    results = search_collection(client, COLLECTION_NAME, query_vector, TOP_K)

    # --- 4. Display Results ---
    if results:
        print(f"\n--- Top {len(results)} Results for: \"{query_text}\" ---\n")
        for i, hit in enumerate(results):
            print(f"Result {i+1} (Score: {hit.score:.4f}):")
            print(f"  Doc ID:  {hit.payload.get('doc_id', 'N/A')}")
            print(f"  Labels:  {hit.payload.get('labels', 'N/A')}")
            print(f"  Summary: {hit.payload.get('summary', 'N/A')}")
            print("-" * 20)
    else:
        logging.warning("No results found or an error occurred during search.")
        
if __name__ == "__main__":
    main()