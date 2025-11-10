import json
import os

# --- Configuration ---
# List of .jsonl files to merge.
# The script will process them in order,
# so results in later files (like the retry file)
# will overwrite results from earlier files.
INPUT_JSONL_FILES = [
    'llm_responses.jsonl',
    'llm_responses_retry.jsonl'
]
OUTPUT_JSON_FILE = 'llm_responses.json' # This will be in your desired Format B
# ---

def convert_jsonl_to_json():
    """
    Reads one or more .jsonl files (one JSON object per line)
    and merges them into a single, large JSON object (Format B).
    
    If a doc_id appears in multiple files, the one from the *last*
    file in the list will be used.
    
    It also filters out any entries that still have errors
    after all retries.
    """
    final_output = {}
    total_lines = 0
    total_errors_overwritten = 0
    final_errors = 0
    
    print(f"Starting conversion of {INPUT_JSONL_FILES} into {OUTPUT_JSON_FILE}...")
    
    for file_path in INPUT_JSONL_FILES:
        if not os.path.exists(file_path):
            print(f"Warning: Input file {file_path} not found. Skipping.")
            continue
            
        print(f"Processing {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f_in:
                for i, line in enumerate(f_in):
                    total_lines += 1
                    try:
                        line_json = json.loads(line.strip())
                        doc_id = list(line_json.keys())[0]
                        result_data = line_json[doc_id]
                        
                        # Check if this doc_id was already processed and failed
                        if doc_id in final_output:
                            existing_entry = final_output[doc_id]
                            if "__PARSE_ERROR__" in existing_entry or "__FATAL_ERROR__" in existing_entry:
                                total_errors_overwritten += 1
                        
                        # Add or overwrite the entry
                        final_output[doc_id] = result_data
                        
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping malformed line {i+1} in {file_path}")
        except Exception as e:
            print(f"An error occurred during reading of {file_path}: {e}")
            return

    # --- Post-processing: Filter out persistent errors ---
    print("\nMerge complete. Filtering out persistent errors...")
    
    final_cleaned_output = {}
    error_output = {}
    
    for doc_id, result_data in final_output.items():
        if "__PARSE_ERROR__" in result_data or "__FATAL_ERROR__" in result_data:
            final_errors += 1
            error_output[doc_id] = result_data
        else:
            final_cleaned_output[doc_id] = result_data

    # --- Write the final, clean JSON file ---
    try:
        with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f_out:
            json.dump(final_cleaned_output, f_out, indent=4)
        
        print("\n--- Summary ---")
        print(f"Successfully converted and cleaned {len(final_cleaned_output)} records.")
        print(f"Final merged JSON (Format B) saved to: {OUTPUT_JSON_FILE}")
        
        if final_errors > 0:
            print(f"Found {final_errors} documents that failed parsing even after retries. See 'failed_docs.json' for details.")
            with open('failed_docs.json', 'w', encoding='utf-8') as f_err:
                json.dump(error_output, f_err, indent=4)
        
    except IOError as e:
        print(f"Error writing final JSON file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during writing: {e}")


if __name__ == "__main__":
    convert_jsonl_to_json()