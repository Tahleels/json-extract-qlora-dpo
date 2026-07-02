# src/deploy/push_to_hub.py
"""
Pushes the GGUF model + LoRA adapters to Hugging Face Hub.
Run: .\\venv\\Scripts\\python -m src.deploy.push_to_hub
You must set HF_TOKEN env var or login via `huggingface-cli login` first.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, upload_file, create_repo

HF_USERNAME = "Tahleels"          # your HF username
REPO_NAME   = "qwen2.5-0.5b-json-extraction-qlora-dpo"
REPO_ID     = f"{HF_USERNAME}/{REPO_NAME}"

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[ERROR] Set HF_TOKEN environment variable first:")
        print("   $env:HF_TOKEN = 'hf_...'")
        sys.exit(1)

    api = HfApi(token=token)

    # Create the repo (private=False = public, anyone can download)
    print(f"Creating repo: {REPO_ID}")
    try:
        create_repo(REPO_ID, repo_type="model", token=token, exist_ok=True)
        print(f"[SUCCESS] Repo ready: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"Repo already exists or error: {e}")

    # Upload GGUF
    artifacts = Path("artifacts")
    gguf_files = list(artifacts.glob("*.gguf")) + list(Path(".").glob("*.gguf"))
    if gguf_files:
        gguf = gguf_files[0]
        print(f"Uploading GGUF: {gguf.name}  ({gguf.stat().st_size / 1e6:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(gguf),
            path_in_repo=f"gguf/{gguf.name}",
            repo_id=REPO_ID,
            token=token,
        )
        print(f"[SUCCESS] GGUF uploaded!")
    else:
        print("[WARNING] No GGUF file found in artifacts/ -- skipping GGUF upload")

    # Upload model card
    card_path = Path("README.md")
    if card_path.exists():
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            token=token,
        )
        print("[SUCCESS] Model card (README.md) uploaded!")

    print(f"\nDone! View your model at:")
    print(f"   https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()
