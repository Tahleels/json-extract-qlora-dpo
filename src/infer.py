import sys
from llama_cpp import Llama
from src.config import load_config

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.infer <input_text>")
        sys.exit(1)
        
    input_text = sys.argv[1]
    cfg = load_config()
    
    # Load model
    import os
    threads = max(1, (os.cpu_count() or 4) - 2)
    llm = Llama(
        model_path="artifacts/model-Q4_K_M.gguf",
        n_ctx=512,
        n_threads=threads,
        verbose=False
    )
    
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
    
    print(output["choices"][0]["text"].strip())

if __name__ == "__main__":
    main()
