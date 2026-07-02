# src/train/export_gguf.py
from __future__ import annotations


def main():
    print("=== GGUF EXPORT STEPS ===")
    print("This script outlines the exact shell commands to run in Colab to convert the merged model to GGUF.")
    print("Make sure you are running in a cell in your Jupyter notebook on Colab:\n")
    
    commands = """
# 1. Clone the llama.cpp repository
git clone https://github.com/ggerganov/llama.cpp.git

# 2. Build llama.cpp (compiles the quantize binary)
make -C llama.cpp

# 3. Install requirements for conversion script
pip install -r llama.cpp/requirements.txt

# 4. Convert the Hugging Face merged model to a float16 GGUF file
python llama.cpp/convert_hf_to_gguf.py artifacts/merged_model --outfile artifacts/model-f16.gguf

# 5. Quantize the GGUF model to 4-bit (Q4_K_M)
./llama.cpp/llama-quantize artifacts/model-f16.gguf artifacts/model-Q4_K_M.gguf Q4_K_M
"""
    print(commands)
    print("=========================")


if __name__ == "__main__":
    main()
