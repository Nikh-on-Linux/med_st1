import ray
import requests
import json
import logging
import sys
import argparse
import time
import re
from tqdm import tqdm

# --- Configuration ---

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s', # Removed problematic 'hostname' key
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Constants
INPUT_FILE = './data/raw.json'
# We will write to a .jsonl file for safety. One JSON object per line.
OUTPUT_FILE = 'llm_responses.jsonl'
SYSTEM_PROMPT = (
    "You are a medical archivist. Your entire response MUST be a single, valid JSON object, and nothing else. "
    "Do not include any explanatory text, markdown, or greetings. "
    "Create a concise abstract for the following document. "
    "This abstract will be used for semantic search. It MUST include: "
    "1) The problem the study is addressing. "
    "2)The methodology used in the study. "
    "3)The key findings (results). "
    "4)The main conclusion or implication. "
    "The final abstract must be a single, coherent paragraph. "
    "The JSON object must contain only one key named 'llm' and the value should be your output."
)
OLLAMA_PORT = 11434
OLLAMA_MODEL = "gpt-oss:20b"
API_TIMEOUT_SECONDS = 300  # 5 minutes
MAX_RETRIES = 1 # Initial try + 1 retry = 2 total attempts

# --- Ray Remote Function (Worker Task) ---

@ray.remote(max_retries=0) # We handle retries manually to log and stop
def process_document(doc_id, document_data):
    """
    This remote function runs on a Ray worker node.
    It processes a single document, calls its local Ollama instance, and returns the result.
    """
    try:
        # 1. Use 127.0.0.1 (localhost) to contact the Ollama instance on this worker
        api_url = f"http://127.0.0.1:{OLLAMA_PORT}/api/generate"

        # 2. Extract data and prepare prompt
        labels = document_data.get("LABELS", [])
        contexts = document_data.get("CONTEXTS", [])
        
        if not contexts:
            logging.warning(f"Document {doc_id} has no CONTEXTS. Skipping.")
            return (doc_id, {"labels": labels, "llm_response": "NO_CONTEXT_PROVIDED"})

        prompt_text = " ".join(contexts)

        # 3. Prepare API Payload
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt_text,
            "system": SYSTEM_PROMPT,
            "stream": False
        }

        # 4. API Call with Retry Logic
        response = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = requests.post(api_url, json=payload, timeout=API_TIMEOUT_SECONDS)
                response.raise_for_status()  # Raise an exception for bad status codes
                
                # If successful, break the loop
                # Ray will automatically prefix this log with the worker IP
                logging.info(f"Successfully processed {doc_id}.")
                break
                
            except requests.exceptions.RequestException as e:
                logging.warning(
                    f"Attempt {attempt + 1} failed for doc {doc_id} on {api_url}: {e}"
                )
                if attempt >= MAX_RETRIES:
                    # This was the last attempt, log error and return a FATAL error to stop the script
                    logging.error(
                        f"Persistent API failure for doc {doc_id} on {api_url} after {MAX_RETRIES + 1} attempts."
                    )
                    # This specific error key will be caught by the main loop to stop the process
                    return (doc_id, {"__FATAL_ERROR__": f"Persistent failure on {doc_id}: {e}"})
                
                # Wait a bit before retrying (simple backoff)
                time.sleep(2 * (attempt + 1))
        
        if response is None:
             # This should not be reachable if logic is correct, but as a safeguard
             return (doc_id, {"__FATAL_ERROR__": f"Unknown failure after retries on {doc_id}."})

        # 5. Parse the successful response
        llm_response = None
        try:
            response_json = response.json()
            inner_json_str = response_json.get("response")
            
            if not inner_json_str:
                raise ValueError("Ollama response field is empty or missing")
                
            try:
                # First, try to parse it directly
                inner_data = json.loads(inner_json_str)
                llm_response = inner_data.get("llm")
            except json.JSONDecodeError as e_parse:
                # If it fails, try to extract it. Find first '{' and last '}'
                logging.warning(f"Response for {doc_id} was not clean JSON. Attempting extraction... Error: {e_parse}")
                
                match = re.search(r'\{.*\}', inner_json_str, re.DOTALL)
                if match:
                    clean_json_str = match.group(0)
                    inner_data = json.loads(clean_json_str)
                    llm_response = inner_data.get("llm")
                else:
                    # This will be caught by the outer Exception
                    raise ValueError(f"Could not find JSON object brackets in response: {inner_json_str[:50]}")

            if llm_response is None:
                # This will be caught by the outer Exception
                raise ValueError(f"Inner JSON response missing 'llm' key even after parsing/extraction. Found: {inner_data.keys()}")

            result_data = {"labels": labels, "llm_response": llm_response}
            return (doc_id, result_data)

        except Exception as e_parse:
            # A parse error is NOT fatal, log it and return a non-fatal error
            logging.error(f"Could not parse/extract JSON for {doc_id}. Full response (truncated): {str(response.text)[:200]}... Error: {e_parse}")
            return (doc_id, {"labels": labels, "llm_response": f"PARSE_ERROR: {e_parse}", "__PARSE_ERROR__": True})

    except Exception as e:
        # Catch any other unexpected errors in the worker
        logging.error(f"Unhandled worker error for doc {doc_id}: {e}")
        # This is likely a coding error, treat as non-fatal for this doc
        return (doc_id, {"labels": [], "llm_response": f"UNHANDLED_WORKER_ERROR: {e}", "__PARSE_ERROR__": True})


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description="Distributed document processing with Ray and Ollama."
    )
    parser.add_argument(
        "--address",
        type=str,
        default=None,
        help="Ray cluster address (e.g., 'ray://<head_node_ip>:10001' or 'auto'). "
             "If not provided, will try to connect to an existing cluster or start a local one."
    )
    args = parser.parse_args()

    try:
        if args.address:
            ray.init(address=args.address, logging_level=logging.ERROR)
        else:
            ray.init(logging_level=logging.ERROR) # Connect to existing or start local
            
        logging.info("Ray cluster connected.")
        logging.info(f"Cluster resources: {ray.cluster_resources()}")
    except Exception as e:
        logging.critical(f"Failed to connect to Ray cluster at {args.address}: {e}")
        logging.critical("Please ensure Ray is running and the address is correct.")
        sys.exit(1)

    # Load the input data
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Loaded {len(data)} documents from {INPUT_FILE}.")
    except FileNotFoundError:
        logging.critical(f"Input file not found: {INPUT_FILE}")
        ray.shutdown()
        sys.exit(1)
    except json.JSONDecodeError:
        logging.critical(f"Could not parse JSON from {INPUT_FILE}.")
        ray.shutdown()
        sys.exit(1)

    # --- Dispatch Tasks ---
    task_refs = []
    for doc_id, document_data in data.items():
        task_refs.append(process_document.remote(doc_id, document_data))

    logging.info(f"Dispatched {len(task_refs)} processing tasks to the Ray cluster.")

    # --- Collect Results and Write to File Incrementally ---
    total_processed = 0
    
    # Use 'w' (write) mode to overwrite any previous partial runs.
    # Change to 'a' (append) mode if you want to resume, but that
    # may result in duplicate entries if you re-run on the same input.
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
            # Use tqdm for a progress bar
            with tqdm(total=len(task_refs), desc="Processing documents") as pbar:
                while task_refs:
                    # ray.wait returns when at least one task is complete
                    ready_refs, remaining_refs = ray.wait(task_refs, num_returns=1)
                    
                    # Get the result from the completed task
                    result_id = ready_refs[0]
                    
                    try:
                        doc_id, result_data = ray.get(result_id)
                    except ray.exceptions.RayTaskError as e:
                        logging.error(f"A worker task crashed unexpectedly: {e}")
                        # Log and continue to the next task
                        total_processed += 1
                        pbar.update(1)
                        task_refs = remaining_refs
                        continue

                    # Update the task list
                    task_refs = remaining_refs

                    # --- Critical Error Check ---
                    # Check if the worker returned a FATAL error object
                    if isinstance(result_data, dict) and "__FATAL_ERROR__" in result_data:
                        logging.critical(
                            f"Fatal error processing doc {doc_id}: {result_data['__FATAL_ERROR__']}\n"
                            "As requested, stopping the entire process."
                        )
                        # Attempt to cancel remaining tasks (best effort)
                        for ref in task_refs:
                            ray.cancel(ref, force=True)
                        
                        ray.shutdown()
                        sys.exit(1)

                    # --- Write Successful or Non-Fatal Error Result to File ---
                    # Format as: {"doc_id": {"labels": [...], "llm_response": "..."}}
                    # This is one valid JSON object per line (jsonl format)
                    output_line = {doc_id: result_data}
                    f_out.write(json.dumps(output_line) + '\n')
                    
                    total_processed += 1
                    pbar.update(1)

    except KeyboardInterrupt:
        logging.warning("User interrupted. Shutting down...")
        for ref in task_refs:
            ray.cancel(ref, force=True)
        ray.shutdown()
        sys.exit(1)
    except IOError as e:
        logging.error(f"Failed to write to output file {OUTPUT_FILE}: {e}")
        # Stop and shut down if we can't write to the file
        for ref in task_refs:
            ray.cancel(ref, force=True)
        ray.shutdown()
        sys.exit(1)


    # Clean up
    ray.shutdown()
    logging.info(f"Processing complete. {total_processed} results saved to {OUTPUT_FILE}.")
    logging.info(f"Run 'python3 convert_jsonl.py' to create the final merged JSON file.")

if __name__ == "__main__":
    main()