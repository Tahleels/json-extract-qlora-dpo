# src/data/validate.py
from __future__ import annotations
import json, sys
from pathlib import Path
from src.schema import is_valid_record, safe_parse


def check_sft(path: Path) -> tuple[int, int]:
    total = bad = 0
    for line in open(path, encoding="utf-8"):
        total += 1
        row = json.loads(line)
        if not is_valid_record(row["output_json"]):
            bad += 1
    return total, bad


def check_dpo(path: Path) -> tuple[int, int]:
    total = bad = 0
    for line in open(path, encoding="utf-8"):
        total += 1
        row = json.loads(line)
        # chosen must be valid; rejected should NOT be a perfect valid record
        if not is_valid_record(row["chosen"]):
            bad += 1
        elif is_valid_record(row["rejected"]) and safe_parse(row["rejected"]) == safe_parse(row["chosen"]):
            bad += 1
    return total, bad


def main():
    data = Path("data")
    ok = True
    for f in ["sft_train.jsonl", "sft_val.jsonl", "sft_test.jsonl"]:
        p = data / f
        if p.exists():
            t, b = check_sft(p)
            print(f"[SFT] {f}: {t} rows, {b} invalid")
            ok &= (b == 0)
    p = data / "dpo_train.jsonl"
    if p.exists():
        t, b = check_dpo(p)
        print(f"[DPO] dpo_train.jsonl: {t} rows, {b} invalid")
        ok &= (b == 0)

    if not ok:
        print("❌ VALIDATION FAILED"); sys.exit(1)
    print("✅ All datasets valid.")


if __name__ == "__main__":
    main()
