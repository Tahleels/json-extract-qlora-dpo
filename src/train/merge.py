# src/train/merge.py
from __future__ import annotations
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import load_config


def main():
    cfg = load_config()
    
    base_model_id = cfg.base_model
    adapter_path = "artifacts/adapter_dpo"
    merged_output_path = "artifacts/merged_model"

    print(f"Loading base model in FP16: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f"Loading DPO adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging weights...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {merged_output_path}")
    merged_model.save_pretrained(merged_output_path)
    tokenizer.save_pretrained(merged_output_path)

    print("Merge complete!")


if __name__ == "__main__":
    main()
