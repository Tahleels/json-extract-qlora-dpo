import json
import os
from tqdm import tqdm
from llama_cpp import Llama
from src.config import load_config
from src.schema import PersonRecord, safe_parse

def evaluate():
    cfg = load_config()
    test_file = "data/sft_test.jsonl"
    
    if not os.path.exists(test_file):
        print(f"Error: Test file {test_file} not found.")
        return
        
    print("Loading GGUF model for evaluation...")
    threads = max(1, (os.cpu_count() or 4) - 2)
    print(f"Using {threads} CPU threads for inference.")
    llm = Llama(
        model_path="artifacts/model-Q4_K_M.gguf",
        n_ctx=512,
        n_threads=threads,
        verbose=False
    )
    
    # Read test set
    with open(test_file, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
        
    print(f"Loaded {len(samples)} test samples.")
    
    valid_json_count = 0
    schema_compliant_count = 0
    markdown_fence_leaks = 0
    total = len(samples)
    
    # Field-level matching stats
    total_fields = 0
    matching_fields = 0
    
    for sample in tqdm(samples, desc="Evaluating"):
        input_text = sample["input_text"]
        expected_json_str = sample["output_json"]
        expected_data = json.loads(expected_json_str)
        
        prompt = (
            f"<|im_start|>system\n{cfg.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{input_text.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        output = llm(
            prompt,
            max_tokens=256,
            stop=["<|im_end|>", "<|endoftext|>"],
            temperature=0.0
        )
        
        raw_output = output["choices"][0]["text"].strip()
        
        # Check for markdown fences leakage
        if "```" in raw_output:
            markdown_fence_leaks += 1
            
        # Parse output
        parsed_record = safe_parse(raw_output)
        from src.schema import is_valid_record
        
        if parsed_record is not None:
            valid_json_count += 1
            
        if is_valid_record(raw_output):
            schema_compliant_count += 1
            
        if parsed_record is not None:
            # Calculate field-level matches
            for k, v in expected_data.items():
                total_fields += 1
                if parsed_record.get(k) == v:
                    matching_fields += 1
        else:
            # If not schema compliant or valid JSON, count all as mismatches
            total_fields += len(expected_data)
            
    # Calculate metrics
    valid_json_pct = (valid_json_count / total) * 100
    schema_compliant_pct = (schema_compliant_count / total) * 100
    markdown_leak_pct = (markdown_fence_leaks / total) * 100
    field_f1 = (matching_fields / total_fields) if total_fields > 0 else 0.0
    
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Total Samples evaluated: {total}")
    print(f"Valid JSON %:           {valid_json_pct:.2f}%")
    print(f"Schema Compliance %:    {schema_compliant_pct:.2f}%")
    print(f"Field-level Accuracy:   {field_f1:.4f}")
    print(f"Markdown Fence Leaks:   {markdown_leak_pct:.2f}% ({markdown_fence_leaks}/{total})")
    print("="*40)

if __name__ == "__main__":
    evaluate()
