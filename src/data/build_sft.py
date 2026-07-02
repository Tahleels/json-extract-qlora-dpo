# src/data/build_sft.py
from __future__ import annotations
import json, random
from pathlib import Path
from src.config import load_config
from src.schema import PersonRecord
from src.data.generate import generate_pairs


def _split(rows, ratios, seed):
    random.seed(seed)
    rows = rows[:]  # copy
    random.shuffle(rows)
    n = len(rows)
    n_tr = int(n * ratios["train"])
    n_va = int(n * ratios["val"])
    return rows[:n_tr], rows[n_tr:n_tr + n_va], rows[n_tr + n_va:]


def to_sft_row(pair: dict) -> dict:
    """SFT training row: instruction/input/output_json (chat-template applied later)."""
    target = PersonRecord(**pair["json"]).to_minified_json()
    return {"input_text": pair["text"], "output_json": target}


def main():
    cfg = load_config()
    pairs = generate_pairs(cfg.data["n_samples"], cfg.seed)
    tr, va, te = _split(pairs, cfg.data["splits"], cfg.seed)

    out = Path(cfg.data["sft_out_dir"]); out.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", tr), ("val", va), ("test", te)]:
        path = out / f"sft_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for p in part:
                f.write(json.dumps(to_sft_row(p)) + "\n")
        print(f"wrote {len(part):>5} rows -> {path}")

    # small committed sample for the repo
    sdir = Path(cfg.data["samples_dir"]); sdir.mkdir(parents=True, exist_ok=True)
    with open(sdir / "sft_sample.jsonl", "w", encoding="utf-8") as f:
        for p in tr[:20]:
            f.write(json.dumps(to_sft_row(p)) + "\n")


if __name__ == "__main__":
    main()
