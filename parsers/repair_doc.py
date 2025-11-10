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
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Constants ---
ORIGINAL_DATA_FILE = './data/raw.json'
FIRST_RUN_OUTPUT_FILE = 'llm_responses.jsonl'
RETRY_OUTPUT_FILE = 'llm_responses_retry.jsonl' # New output file for retries

# Copy the exact same system prompt from the main script
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
# This is the exact same function as in the main script.
# It's good practice to keep it identical or import it from a shared file.
@ray.remote(max_retries=0) 
def process_document(doc_id, document_data):
    """
    This remote function runs on a Ray worker node.
    It processes a single document, calls its local Ollama instance, and returns the result.
    """
    try:
        api_url = f"http://127.0.0.1:{OLLAMA_PORT}/api/generate"
        labels = document_data.get("LABELS", [])
        contexts = document_data.get("CONTEXTS", [])
        
        if not contexts:
            logging.warning(f"Document {doc_id} has no CONTEXTS. Skipping.")
            return (doc_id, {"labels": labels, "llm_response": "NO_CONTEXT_PROVIDED"})

        prompt_text = " ".join(contexts)

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt_text,
            "system": SYSTEM_PROMPT,
            "stream": False
        }

        response = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = requests.post(api_url, json=payload, timeout=API_TIMEOUT_SECONDS)
                response.raise_for_status()
                logging.info(f"[RETRY] Successfully processed {doc_id}.")
                break
            except requests.exceptions.RequestException as e:
                logging.warning(
                    f"[RETRY] Attempt {attempt + 1} failed for doc {doc_id} on {api_url}: {e}"
                )
                if attempt >= MAX_RETRIES:
                    logging.error(
                        f"[RETRY] Persistent API failure for doc {doc_id} on {api_url} after {MAX_RETRIES + 1} attempts."
                    )
                    return (doc_id, {"__FATAL_ERROR__": f"Persistent failure on {doc_id}: {e}"})
                time.sleep(2 * (attempt + 1))
        
        if response is None:
             return (doc_id, {"__FATAL_ERROR__": f"Unknown failure after retries on {doc_id}."})

        llm_response = None
        try:
            response_json = response.json()
            inner_json_str = response_json.get("response")
            
            if not inner_json_str:
                raise ValueError("Ollama response field is empty or missing")
                
            try:
                inner_data = json.loads(inner_json_str)
                llm_response = inner_data.get("llm")
            except json.JSONDecodeError as e_parse:
                logging.warning(f"[RETRY] Response for {doc_id} was not clean JSON. Attempting extraction... Error: {e_parse}")
                match = re.search(r'\{.*\}', inner_json_str, re.DOTALL)
                if match:
                    clean_json_str = match.group(0)
                    inner_data = json.loads(clean_json_str)
                    llm_response = inner_data.get("llm")
                else:
                    raise ValueError(f"Could not find JSON object brackets in response: {inner_json_str[:50]}")

            if llm_response is None:
                raise ValueError(f"Inner JSON response missing 'llm' key even after parsing/extraction. Found: {inner_data.keys()}")

            result_data = {"labels": labels, "llm_response": llm_response}
            return (doc_id, result_data)

        except Exception as e_parse:
            logging.error(f"[RETRY] Could not parse/extract JSON for {doc_id}. Full response (truncated): {str(response.text)[:200]}... Error: {e_parse}")
            return (doc_id, {"labels": labels, "llm_response": f"PARSE_ERROR: {e_parse}", "__PARSE_ERROR__": True})

    except Exception as e:
        logging.error(f"[RETRY] Unhandled worker error for doc {doc_id}: {e}")
        return (doc_id, {"labels": [], "llm_response": f"UNHANDLED_WORKER_ERROR: {e}", "__PARSE_ERROR__": True})


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description="Retry failed documents from a previous processing run."
    )
    parser.add_argument(
        "--address",
        type=str,
        default=None,
        help="Ray cluster address (e.g., 'ray://<head_node_ip>:10001' or 'auto')."
    )
    args = parser.parse_args()

    # --- Step 1: Find failed doc IDs from the first run ---
    logging.info(f"Checking {FIRST_RUN_OUTPUT_FILE} for failed documents...")
    failed_doc_ids = []
    try:
        with open(FIRST_RUN_OUTPUT_FILE, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                try:
                    line_json = json.loads(line.strip())
                    doc_id = list(line_json.keys())[0]
                    result_data = line_json[doc_id]
                    
                    # Check for either a parse error or a fatal connection error
                    if "__PARSE_ERROR__" in result_data or "__FATAL_ERROR__" in result_data:
                        failed_doc_ids.append(doc_id)
                except Exception as e:
                    logging.warning(f"Skipping malformed line in {FIRST_RUN_OUTPUT_FILE}: {e}")
                    
    except FileNotFoundError:
        logging.critical(f"Input file not found: {FIRST_RUN_OUTPUT_FILE}. Cannot determine which documents to retry.")
        sys.exit(1)

    if not failed_doc_ids:
        logging.info("No failed documents found. Exiting.")
        sys.exit(0)

    logging.info(f"Found {len(failed_doc_ids)} documents to retry.")

    # --- Step 2: Load original data for the failed documents ---
    try:
        with open(ORIGINAL_DATA_FILE, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        data_to_retry = {}
        for doc_id in failed_doc_ids:
            if doc_id in original_data:
                data_to_retry[doc_id] = original_data[doc_id]
            else:
                logging.warning(f"Doc ID {doc_id} from logs not found in {ORIGINAL_DATA_FILE}. Skipping.")
                
    except FileNotFoundError:
        logging.critical(f"Original data file not found: {ORIGINAL_DATA_FILE}")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.critical(f"Could not parse JSON from {ORIGINAL_DATA_FILE}.")
        sys.exit(1)

    if not data_to_retry:
        logging.info("No valid documents to retry. Exiting.")
        sys.exit(0)

    # --- Step 3: Connect to Ray and re-run tasks ---
    try:
        if args.address:
            ray.init(address=args.address, logging_level=logging.ERROR)
        else:
            ray.init(logging_level=logging.ERROR)
        logging.info("Ray cluster connected for retry run.")
    except Exception as e:
        logging.critical(f"Failed to connect to Ray cluster at {args.address}: {e}")
        sys.exit(1)

    task_refs = []
    for doc_id, document_data in data_to_retry.items():
        task_refs.append(process_document.remote(doc_id, document_data))

    logging.info(f"Dispatched {len(task_refs)} retry tasks to the Ray cluster.")

    # --- Step 4: Collect retry results and write to new file ---
    total_retried = 0
    try:
        with open(RETRY_OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
            with tqdm(total=len(task_refs), desc="Retrying documents") as pbar:
                while task_refs:
                    ready_refs, remaining_refs = ray.wait(task_refs, num_returns=1)
                    result_id = ready_refs[0]
                    
                    try:
                        doc_id, result_data = ray.get(result_id)
                    except ray.exceptions.RayTaskError as e:
                        logging.error(f"[RETRY] A worker task crashed unexpectedly: {e}")
                        total_retried += 1
                        pbar.update(1)
                        task_refs = remaining_refs
                        continue

                    task_refs = remaining_refs

                    # Check for fatal error again
                    if isinstance(result_data, dict) and "__FATAL_ERROR__" in result_data:
                        logging.critical(
                            f"[RETRY] Fatal error processing doc {doc_id}: {result_data['__FATAL_ERROR__']}\n"
                            "Stopping the retry process."
                        )
                        for ref in task_refs:
                            ray.cancel(ref, force=True)
                        ray.shutdown()
                        sys.exit(1)

                    # Write the new result (even if it's another parse error)
                    output_line = {doc_id: result_data}
                    f_out.write(json.dumps(output_line) + '\n')
                    
                    total_retried += 1
                    pbar.update(1)

    except KeyboardInterrupt:
        logging.warning("User interrupted retry run. Shutting down...")
        for ref in task_refs:
            ray.cancel(ref, force=True)
    except IOError as e:
        logging.error(f"Failed to write to retry output file {RETRY_OUTPUT_FILE}: {e}")
        for ref in task_refs:
            ray.cancel(ref, force=True)

    ray.shutdown()
    logging.info(f"Retry process complete. {total_retried} results saved to {RETRY_OUTPUT_FILE}.")
    logging.info(f"Run 'python3 convert_jsonl.py' to create the final merged JSON file.")

if __name__ == "__main__":
    main()