# src/train/train_dpo.py
from __future__ import annotations
import torch
import yaml
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from trl import DPOTrainer, DPOConfig
from src.config import load_config


def main():
    cfg = load_config()

    # Load DPO config
    with open("configs/dpo.yaml", "r", encoding="utf-8") as f:
        dpo_cfg = yaml.safe_load(f)

    # 4-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model: {cfg.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load base model in 4-bit NF4
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Load SFT adapter on top, set is_trainable=True
    sft_adapter_path = "artifacts/adapter_sft"
    print(f"Loading SFT adapter from: {sft_adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        sft_adapter_path,
        is_trainable=True,
    )

    # Load DPO dataset
    dpo_dataset_path = Path(cfg.data["dpo_out_dir"]) / "dpo_train.jsonl"
    print(f"Loading DPO dataset: {dpo_dataset_path}")
    dataset = load_dataset("json", data_files={"train": str(dpo_dataset_path)})["train"]

    # Format prompt, chosen, rejected with chat template
    def format_dpo(example):
        messages = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": example["prompt"]},
        ]
        # Generation prompt adds assistant prefix
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {
            "prompt": prompt_text,
            "chosen": example["chosen"] + tokenizer.eos_token,
            "rejected": example["rejected"] + tokenizer.eos_token,
        }

    dpo_dataset = dataset.map(format_dpo)

    # Setup DPOTrainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None, # TRL will automatically handle ref model by disabling adapters
        train_dataset=dpo_dataset,
        tokenizer=tokenizer,
        args=DPOConfig(
            output_dir=dpo_cfg["training"]["output_dir"],
            beta=dpo_cfg["training"]["beta"],
            num_train_epochs=dpo_cfg["training"]["epochs"],
            per_device_train_batch_size=dpo_cfg["training"]["batch_size"],
            gradient_accumulation_steps=dpo_cfg["training"]["grad_accum"],
            learning_rate=float(dpo_cfg["training"]["learning_rate"]),
            max_length=dpo_cfg["training"]["max_seq_length"],
            max_prompt_length=256,
            logging_steps=10,
            save_strategy="epoch",
            fp16=True,
            report_to="none",
        ),
    )

    print("Starting DPO training...")
    trainer.train()

    print(f"Saving DPO adapter to {dpo_cfg['training']['output_dir']}")
    trainer.save_model(dpo_cfg["training"]["output_dir"])
    print("DPO complete!")


if __name__ == "__main__":
    main()
