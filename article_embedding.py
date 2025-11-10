import ray
import json
import logging
import sys
import argparse
import time
import math
import uuid  # Import the UUID library
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct

# --- Configuration ---

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- File and Qdrant Configuration ---
RESPONSES_FILE = 'llm_responses.json'
QDRANT_HOST = '172.25.120.212'
QDRANT_PORT = 6333  # User-specified REST port
COLLECTION_NAME = 'med_st1'
MODEL_NAME = 'ncbi/MedCPT-Article-Encoder' # Use the Article Encoder for documents
MODEL_DIMENSION = 768
BATCH_SIZE = 64 # How many docs to send to each worker at a time

# --- Ray Worker Actor ---

@ray.remote(num_gpus=1)  # Assign one GPU to each actor
class EmbeddingWorker:
    """
    A Ray actor that loads the MedCPT model onto a GPU once and
    processes batches of documents to embed and upload them to Qdrant.
    
    This worker now uses the 'transformers' library directly as recommended.
    """
    def __init__(self, qdrant_host, qdrant_port, collection_name):
        # This code runs once when the actor is created
        import torch
        from transformers import AutoTokenizer, AutoModel
        from qdrant_client import QdrantClient
        
        # 1. Load the model and tokenizer onto the GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Worker actor initializing model on device: {self.device}")
        
        # Load model and tokenizer using transformers library
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval() # Set model to evaluation mode
        
        logging.info(f"Worker actor model {MODEL_NAME} loaded.")
        
        # 2. Connect to the central Qdrant server from the worker
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        
        # Initialize client here. It will be reused.
        try:
            self.qdrant_client = QdrantClient(
                host=self.qdrant_host, 
                port=self.qdrant_port
            )
            # Use a valid lightweight call to check connection
            self.qdrant_client.get_collections() # Use as a valid health check
            logging.info(f"Worker actor connected to Qdrant at {qdrant_host}:{qdrant_port}")
        except Exception as e:
            logging.error(f"Worker actor FAILED to connect to Qdrant: {e}")
            # If we can't connect, this actor is useless. Raise an error.
            raise ConnectionError(f"Worker could not connect to Qdrant: {e}")

    def _get_cls_embedding(self, model_output, attention_mask):
        """
        MedCPT uses the [CLS] token embedding.
        This is a helper to extract it.
        """
        # We want the last_hidden_state
        # It has shape [batch_size, sequence_length, hidden_dim]
        # We just want the embedding for the first token, [CLS], which is at index 0.
        cls_embedding = model_output.last_hidden_state[:, 0]
        return cls_embedding

    def embed_and_upload(self, batch):
        """
        Takes a batch of documents, generates embeddings, and uploads to Qdrant.
        """
        import torch

        # Handle the empty batch from the init task
        if not batch:
            return 0
            
        try:
            # 1. Prepare texts for embedding (the LLM-generated summaries)
            texts_to_embed = [item['summary'] for item in batch]
            
            # 2. Generate embeddings
            with torch.no_grad():
                # Tokenize the batch
                inputs = self.tokenizer(
                    texts_to_embed, 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt', 
                    max_length=512
                ).to(self.device)
                
                # Get model outputs
                outputs = self.model(**inputs)
                
                # Get the [CLS] embedding (as required by MedCPT)
                embeddings_tensor = self._get_cls_embedding(outputs, inputs['attention_mask'])
                
                # Move embeddings to CPU and convert to numpy array
                embeddings = embeddings_tensor.cpu().numpy()

            
            # 3. Prepare Qdrant points with metadata
            points = []
            for i, item in enumerate(batch):
                doc_id = item['doc_id']
                vector = embeddings[i].tolist()
                
                # As requested: doc_id, labels, and the summary (llm_response)
                payload = {
                    "doc_id": doc_id, # Store the original doc_id in the payload
                    "labels": item['labels'],
                    "summary": item['summary'] # Store the summary we embedded
                }
                
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),  # Generate a new, valid UUID for the point ID
                        vector=vector,
                        payload=payload
                    )
                )

            # 4. Upload the batch to Qdrant
            if points:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True  # Wait for the operation to complete
                )
            return len(points) # Return the number of processed items
            
        except Exception as e:
            logging.error(f"Worker failed to embed/upload batch. Error: {e}")
            return 0 # Return 0 successes for this batch

# --- Main Execution ---

def setup_qdrant_collection(client, collection_name, vector_size):
    """
    Checks if the Qdrant collection exists and creates it if not.
    """
    try:
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            logging.info(f"Collection '{collection_name}' not found. Creating it...")
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, 
                    distance=models.Distance.COSINE
                )
            )
            logging.info(f"Collection '{collection_name}' created successfully.")
        else:
            logging.info(f"Collection '{collection_name}' already exists.")
            
    except Exception as e:
        logging.critical(f"Failed to setup Qdrant collection: {e}")
        logging.critical(f"Please ensure Qdrant server is running at {QDRANT_HOST}:{QDRANT_PORT} and is accessible.")
        sys.exit(1)

def load_and_prepare_data():
    """
    Loads the llm_responses.json file and filters out failed documents.
    """
    logging.info(f"Loading summaries from {RESPONSES_FILE}...")
    try:
        with open(RESPONSES_FILE, 'r', encoding='utf-8') as f:
            responses = json.load(f)
    except Exception as e:
        logging.critical(f"Failed to load {RESPONSES_FILE}: {e}")
        sys.exit(1)
        
    logging.info("Preparing documents for embedding...")
    docs_to_process = []
    processed_count = 0
    failed_count = 0
    
    for doc_id, data in responses.items():
        llm_response = data.get('llm_response', '')
        
        # Skip documents that had errors in the previous step
        if "PARSE_ERROR" in llm_response or "UNHANDLED_WORKER_ERROR" in llm_response:
            failed_count += 1
            continue
            
        # Skip documents with no summary
        if not llm_response:
            failed_count += 1
            continue
            
        # Add to the processing list
        docs_to_process.append({
            "doc_id": doc_id,
            "summary": llm_response,
            "labels": data.get("labels", [])
        })
        processed_count += 1

    logging.info(f"Prepared {processed_count} valid documents for embedding.")
    if failed_count > 0:
        logging.warning(f"Skipped {failed_count} documents due to errors or missing data in {RESPONSES_FILE}.")
        
    return docs_to_process

def main():
    parser = argparse.ArgumentParser(description="Embed and store documents in Qdrant using Ray.")
    parser.add_argument(
        "--address",
        type=str,
        default=None,
        help="Ray cluster address (e.g., 'auto' or 'ray://<head_node_ip>:10001')."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel embedding workers (actors) to create. Should be <= number of GPUs."
    )
    args = parser.parse_args()

    # --- 1. Connect to Ray ---
    try:
        if args.address:
            ray.init(address=args.address, logging_level=logging.ERROR)
        else:
            ray.init(logging_level=logging.ERROR)
        logging.info("Ray cluster connected.")
        
        # Check if we have enough GPUs for the requested workers
        available_gpus = ray.cluster_resources().get("GPU", 0)
        if args.num_workers > available_gpus:
            logging.warning(
                f"Requested {args.num_workers} workers but only {available_gpus} GPUs are available."
            )
            args.num_workers = int(available_gpus)
            if args.num_workers == 0:
                logging.error("No GPUs available in the cluster. Exiting.")
                logging.error("Set num_gpus=0 in the @ray.remote decorator to run on CPU (very slow).")
                ray.shutdown()
                sys.exit(1)
            logging.warning(f"Proceeding with {args.num_workers} workers.")
            
    except Exception as e:
        logging.critical(f"Failed to connect to Ray: {e}")
        sys.exit(1)

    # --- 2. Load and Prepare Data ---
    docs_to_process = load_and_prepare_data()
    if not docs_to_process:
        logging.info("No documents to process. Exiting.")
        ray.shutdown()
        sys.exit(0)

    # --- 3. Setup Qdrant Collection (on main thread) ---
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        # Use get_collections() as a valid health check
        client.get_collections() 
        logging.info(f"Successfully connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}.")
        setup_qdrant_collection(client, COLLECTION_NAME, MODEL_DIMENSION)
    except Exception as e:
        logging.critical(f"Failed to connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}. Error: {e}")
        ray.shutdown()
        sys.exit(1)

    # --- 4. Create Ray Actor Pool ---
    logging.info(f"Creating a pool of {args.num_workers} embedding workers...")
    try:
        worker_pool = [
            EmbeddingWorker.remote(QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME) 
            for _ in range(args.num_workers)
        ]
        # Wait for all actors to initialize
        init_tasks = [actor.embed_and_upload.remote([]) for actor in worker_pool]
        ray.get(init_tasks) # This confirms they are all loaded and connected
        logging.info("All workers are initialized and connected to Qdrant.")
    except Exception as e:
        logging.critical(f"Failed to create worker pool. An actor may have failed to start. Error: {e}")
        ray.shutdown()
        sys.exit(1)

    # --- 5. Dispatch Batches to Workers ---
    batches = [
        docs_to_process[i:i + BATCH_SIZE] 
        for i in range(0, len(docs_to_process), BATCH_SIZE)
    ]
    logging.info(f"Divided {len(docs_to_process)} documents into {len(batches)} batches.")

    task_refs = []
    actor_index = 0
    for batch in batches:
        # Round-robin distribution of batches to the worker actors
        actor = worker_pool[actor_index]
        task_refs.append(actor.embed_and_upload.remote(batch))
        actor_index = (actor_index + 1) % args.num_workers

    # --- 6. Track Progress ---
    total_embedded = 0
    with tqdm(total=len(docs_to_process), desc="Embedding and Uploading") as pbar:
        while task_refs:
            ready_refs, remaining_refs = ray.wait(task_refs, num_returns=1)
            
            try:
                num_processed = ray.get(ready_refs[0])
                total_embedded += num_processed
                pbar.update(num_processed)
            except ray.exceptions.RayTaskError as e:
                logging.error(f"A worker task crashed while processing a batch: {e}")
            
            task_refs = remaining_refs

    ray.shutdown()
    logging.info("--- Embedding and Upload Complete ---")
    logging.info(f"Successfully embedded and uploaded {total_embedded} documents to Qdrant.")
    logging.info(f"Collection '{COLLECTION_NAME}' at {QDRANT_HOST}:{QDRANT_PORT} is ready.")

if __name__ == "__main__":
    main()