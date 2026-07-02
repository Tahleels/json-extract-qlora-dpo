# src/train/train_sft.py
from __future__ import annotations
import torch
import yaml
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from src.config import load_config


def main():
    cfg = load_config()
    
    # Load SFT config
    with open("configs/sft.yaml", "r", encoding="utf-8") as f:
        sft_cfg = yaml.safe_load(f)

    # 4-bit Quantization Config (NF4)
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

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # LoRA Adapter Configuration
    peft_config = LoraConfig(
        r=sft_cfg["lora"]["r"],
        lora_alpha=sft_cfg["lora"]["alpha"],
        lora_dropout=sft_cfg["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=sft_cfg["lora"]["target_modules"],
    )

    # Load SFT datasets (local JSONL files)
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(Path(cfg.data["sft_out_dir"]) / "sft_train.jsonl"),
            "val": str(Path(cfg.data["sft_out_dir"]) / "sft_val.jsonl"),
        }
    )

    # Format into Qwen-Style Chat Template
    def format_chat(example):
        messages = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": example["input_text"]},
            {"role": "assistant", "content": example["output_json"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    train_ds = dataset["train"].map(format_chat)
    val_ds = dataset["val"].map(format_chat)

    # Setup Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        dataset_text_field="text",
        args=SFTConfig(
            output_dir=sft_cfg["training"]["output_dir"],
            num_train_epochs=sft_cfg["training"]["epochs"],
            per_device_train_batch_size=sft_cfg["training"]["batch_size"],
            gradient_accumulation_steps=sft_cfg["training"]["grad_accum"],
            learning_rate=float(sft_cfg["training"]["learning_rate"]),
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="epoch",
            fp16=True,
            max_seq_length=sft_cfg["training"]["max_seq_length"],
            report_to="none",
        ),
    )

    print("Starting SFT training...")
    trainer.train()

    print(f"Saving SFT adapter to {sft_cfg['training']['output_dir']}")
    trainer.save_model(sft_cfg["training"]["output_dir"])
    print("SFT complete!")


if __name__ == "__main__":
    main()
