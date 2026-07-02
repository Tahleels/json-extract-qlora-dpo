# src/data/build_dpo.py
from __future__ import annotations
import json
from pathlib import Path
from src.config import load_config
from src.schema import PersonRecord
from src.data.generate import generate_pairs
from src.data.corrupt import make_rejected


def main():
    cfg = load_config()
    # reuse the same generator but a different seed slice so DPO != SFT text
    pairs = generate_pairs(cfg.data["n_samples"] // 2, cfg.seed + 1)

    out = Path(cfg.data["dpo_out_dir"]); out.mkdir(parents=True, exist_ok=True)
    path = out / "dpo_train.jsonl"
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for i, p in enumerate(pairs):
            clean = PersonRecord(**p["json"]).to_minified_json()
            rejected = make_rejected(clean, seed=cfg.seed + i)
            if rejected == clean:      # guarantee they differ
                continue
            row = {
                # DPOTrainer expects prompt/chosen/rejected
                "prompt": p["text"],
                "chosen": clean,
                "rejected": rejected,
            }
            f.write(json.dumps(row) + "\n")
            n += 1
    print(f"wrote {n} DPO pairs -> {path}")

    sdir = Path(cfg.data["samples_dir"]); sdir.mkdir(parents=True, exist_ok=True)
    with open(sdir / "dpo_sample.jsonl", "w", encoding="utf-8") as f:
        for p in pairs[:20]:
            clean = PersonRecord(**p["json"]).to_minified_json()
            f.write(json.dumps({"prompt": p["text"], "chosen": clean,
                                "rejected": make_rejected(clean)}) + "\n")


if __name__ == "__main__":
    main()
