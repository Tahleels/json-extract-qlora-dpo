# Codebase Map — What Every File Does

> Quick reference for navigating this repository. Every file has one job.

---

## Root

| File | Purpose |
|---|---|
| `README.md` | Public-facing model card — shown on GitHub and Hugging Face Hub |
| `CODEBASE.md` | This file — internal developer map of the repo |
| `INTERVIEWER.md` | Deep-dive project explanation with design decisions and tradeoffs |
| `.gitignore` | Excludes `venv/`, `data/*.jsonl`, `artifacts/*.gguf`, `__pycache__` from Git |

---

## configs/

All hyperparameters are in YAML files — not hardcoded. This makes experiments easy to re-run.

| File | Purpose |
|---|---|
| `base.yaml` | Shared config — model ID (`Qwen/Qwen2.5-0.5B-Instruct`), schema field names, dataset sizes (3400 train / 200 val / 400 test / 2000 DPO pairs) |
| `sft.yaml` | SFT-specific hyperparameters — LoRA rank (16), alpha (32), learning rate (2e-4), epochs (3), batch size, target modules |
| `dpo.yaml` | DPO-specific hyperparameters — beta (0.1), learning rate (5e-6), epochs (1), batch size |

---

## src/

### Core files (root of `src/`)

| File | Purpose |
|---|---|
| `schema.py` | Defines `PersonRecord` — the Pydantic model that enforces the JSON schema (`name`, `age`, `job_title`, `company`, `city`, `email`). Also contains `is_valid_record()` and `safe_parse()` validators used across the whole project |
| `config.py` | Loads all YAML config files and merges them into a single Python dict. Every other script calls this instead of reading YAML directly |
| `infer.py` | Local CPU inference entry point. Takes a text string, loads the GGUF model via `llama-cpp-python`, builds the chat prompt, and prints the extracted JSON. Run as: `python -m src.infer "..."` |

---

### src/data/

Everything about generating and preparing training data.

| File | Purpose |
|---|---|
| `generate.py` | Generates synthetic `PersonRecord` data using the `Faker` library across 14 sentence templates. Each call produces one `(input_text, output_json)` pair |
| `validate.py` | Runs schema validation on generated data using `safe_parse()` to catch any malformed samples before they enter training |
| `corrupt.py` | Takes a clean JSON string and deliberately corrupts it in one of 5 ways: adds markdown fences, adds a chatty preamble, adds trailing commas, uses single quotes, or removes a random field. Used to build DPO "rejected" samples |
| `build_sft.py` | Generates and writes the SFT dataset (`data/sft_train.jsonl`, `sft_val.jsonl`, `sft_test.jsonl`) in the chat-template format ready for `SFTTrainer` |
| `build_dpo.py` | Generates and writes the DPO dataset (`data/dpo_train.jsonl`) — each row has a `prompt`, a `chosen` (clean JSON), and a `rejected` (corrupted JSON) output |
| `push_dataset.py` | Pushes the generated datasets to the Hugging Face Datasets Hub |

---

### src/train/

Everything about training the model. Designed to run on a free Colab T4 GPU.

| File | Purpose |
|---|---|
| `train_sft.py` | Loads the base Qwen2.5-0.5B model in 4-bit NF4, attaches LoRA adapters to all attention and MLP projection layers, and runs `SFTTrainer` from TRL. Saves adapter to `adapter_sft/` |
| `train_dpo.py` | Loads the SFT-trained adapter as the "reference model" and trains a second LoRA on top using `DPOTrainer` with the preference pairs from `dpo_train.jsonl`. Saves adapter to `adapter_dpo/` |
| `merge.py` | Merges the DPO LoRA adapter weights back into the base model to produce a single, self-contained model (no adapter dependency). Required before GGUF export |
| `export_gguf.py` | Calls `llama.cpp`'s `convert_hf_to_gguf.py` script and then `llama-quantize` to produce the final `model-Q4_K_M.gguf` artifact — ~398MB, runs on any CPU laptop |

---

### src/eval/

| File | Purpose |
|---|---|
| `evaluate_gguf.py` | Loads the GGUF model and runs inference on all 400 held-out test samples. Reports: Valid JSON %, Schema Compliance %, Field-level Accuracy, and Markdown Fence Leak %. Used to produce the numbers in the README |

---

### src/deploy/

| File | Purpose |
|---|---|
| `app.py` | Gradio web app for the Hugging Face Space. On startup it downloads the GGUF from the HF model repo, loads it via `llama-cpp-python`, and exposes a simple text-in → JSON-out UI with 3 example inputs and live validation feedback |
| `push_to_hub.py` | One-shot upload script — creates the HF model repo (if needed) and uploads the GGUF file and README. Run once after training with your `HF_TOKEN` set |

---

## notebooks/

| File | Purpose |
|---|---|
| `colab_end_to_end.ipynb` | The single source of truth for training. Self-contained Colab notebook that: installs libraries, clones the repo, generates data, runs SFT, runs DPO, merges, and exports to GGUF — all in one session on a free T4 GPU |

---

## artifacts/

| File | Purpose |
|---|---|
| `model-Q4_K_M.gguf` | The final trained model in GGUF Q4_K_M format (~398MB). Git-ignored (too large). Produced by the Colab notebook and downloaded manually for local evaluation |

---

## data/

| Path | Purpose |
|---|---|
| `data/samples/` | A small committed sample of each dataset (10 rows each) for repo inspection — the full JSONL files are git-ignored |
| `data/sft_train.jsonl` etc. | Full generated datasets — git-ignored, regenerated by running `build_sft.py` and `build_dpo.py` |
