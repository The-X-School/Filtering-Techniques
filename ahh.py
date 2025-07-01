import json
import re
from tinyagent.schema import Tool # Only Tool is needed from schema

# Define your core logic as a regular function
def _extract_filtered_data_logic(text: str):
    """
    Extracts summary, key terms, and document type from raw text using simple heuristics.
    This is the core logic that will be wrapped by the Tool.
    """
    summary = text[:200].replace('\n', ' ').strip()
    key_terms = list(set(re.findall(r'\b[A-Z][a-zA-Z]{3,}\b', text)))[:10]
    
    if "Calculus" in text or "Math" in text:
        doc_type = "mathematics"
    elif "business" in text.lower():
        doc_type = "business"
    else:
        doc_type = "general"
    
    return {
        "summary": summary,
        "key_terms": key_terms,
        "document_type": doc_type
    }

# Create a concrete subclass of Tool that implements the _run method
class CustomExtractDataTool(Tool):
    # In the __init__, call the parent Tool's constructor to set up metadata
    def __init__(self):
        super().__init__(
            name="extract_filtered_data",
            args={"text": str}, # This describes the arguments the tool expects
            description="Extracts summary, key terms, and document type from raw text"
            # Do NOT pass func= here. We implement _run directly below.
        )
    
    # Implement the abstract _run method which the Tool class requires
    # This method is what tinyagent's Agent would call to execute your tool.
    def _run(self, text: str):
        # Call your core logic function here
        return _extract_filtered_data_logic(text)

# Instantiate your custom tool class
extract_filtered_data_tool = CustomExtractDataTool()


# --- Rest of your processing code (remains largely the same) ---

# Path to input file
input_path = "climblab_samples/00001.jsonl"
output_path = "data/filtered_dataset/tinyagent_output.jsonl" 

# Ensure output directory exists
import os
os.makedirs(os.path.dirname(output_path), exist_ok=True)


# Process each entry
try:
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        
        print(f"Processing data from '{input_path}'...")
        processed_count = 0
        for line in infile:
            if not line.strip():
                continue  # skip empty lines

            try:
                data = json.loads(line)
                text = data.get("text", "")

                if not text:
                    print(f"Warning: 'text' field missing or empty in line: {line.strip()}")
                    continue

                # Call the _run method of the Tool instance
                # This is the standard way to execute a tinyagent Tool directly
                extracted_data = extract_filtered_data_tool._run(text=text) # Pass args as keywords

                output = {
                    "input_text": text,
                    "extracted_data": extracted_data
                }

                outfile.write(json.dumps(output) + "\n")
                processed_count += 1

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in line: {line.strip()} - {e}")
            except Exception as e:
                print(f"An error occurred while processing line: {line.strip()} - {e}")

    print(f"\nProcessing complete! Successfully processed {processed_count} entries.")
    print(f"Output saved to: '{output_path}'")

except FileNotFoundError:
    print(f"Error: Input file not found at '{input_path}'. Please ensure the file exists.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")