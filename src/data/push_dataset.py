# src/data/push_dataset.py
from __future__ import annotations
from datasets import load_dataset, DatasetDict
from src.config import load_config


def main():
    cfg = load_config()
    ds = DatasetDict({
        "sft_train": load_dataset("json", data_files="data/sft_train.jsonl")["train"],
        "sft_val":   load_dataset("json", data_files="data/sft_val.jsonl")["train"],
        "sft_test":  load_dataset("json", data_files="data/sft_test.jsonl")["train"],
        "dpo_train": load_dataset("json", data_files="data/dpo_train.jsonl")["train"],
    })
    ds.push_to_hub(cfg.dataset_repo, private=False)
    print(f"✅ pushed to https://huggingface.co/datasets/{cfg.dataset_repo}")


if __name__ == "__main__":
    # run `huggingface-cli login` first
    main()
