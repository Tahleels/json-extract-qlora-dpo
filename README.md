# JSON Extraction LLM — QLoRA SFT + DPO

> Fine-tuned **Qwen2.5-0.5B-Instruct** for reliable structured JSON extraction using **QLoRA + DPO** on a free Colab T4 GPU. Quantized to **GGUF Q4_K_M** — runs locally on any CPU laptop.

[![HF Model](https://img.shields.io/badge/🤗%20Model-Tahleels%2Fqwen2.5--0.5b--json--extraction-blue)](https://huggingface.co/Tahleels/qwen2.5-0.5b-json-extraction-qlora-dpo)
[![HF Space](https://img.shields.io/badge/🤗%20Demo-Gradio%20Space-orange)](https://huggingface.co/spaces/Tahleels/json-extraction-demo)
[![GitHub](https://img.shields.io/badge/GitHub-Tahleels%2Fjson--extract--qlora--dpo-black)](https://github.com/Tahleels/json-extract-qlora-dpo)

---

## What This Does

Given unstructured natural-language text, the model outputs a **strict, schema-conforming JSON object** — nothing else.

**Input:**
```
Hi, I'm Sarah Chen, 34. I work as a Senior Data Analyst at TechCorp in Seattle.
Reach me at sarah.chen@techcorp.io.
```

**Output:**
```json
{"age":34,"city":"Seattle","company":"TechCorp","email":"sarah.chen@techcorp.io","job_title":"Senior Data Analyst","name":"Sarah Chen"}
```

No markdown fences. No preamble. No extra text. Just valid JSON.

---

## Architecture

```
LOCAL (CPU laptop)
├── Data generation + cleaning (Faker, Pydantic)
├── Eval harness (runs GGUF via llama-cpp-python)
└── Git + config management
        │
        ▼  push to GitHub
COLAB FREE T4 (16GB VRAM)
├── Base: Qwen2.5-0.5B-Instruct
├── Stage A: QLoRA SFT (TRL SFTTrainer, 4-bit NF4)
│           └── adapter_sft/
├── Stage B: DPO on SFT model (TRL DPOTrainer)
│           └── adapter_dpo/
└── Merge → GGUF Q4_K_M (~400MB)
        │
        ▼
DEPLOY (free HF hosting)
├── HF Hub: model card + GGUF file
└── HF Spaces: Gradio demo (CPU basic tier)
```

---

## Results

| Metric | Base (Qwen2.5-0.5B) | + SFT + DPO (this model) |
|---|---|---|
| Valid JSON % | ~62% | **100.00%** |
| Schema compliance % | ~40% | **100.00%** |
| Field-level Accuracy | ~0.55 | **0.9996 (99.96%)** |
| Markdown fence leaks % | ~55% | **0.00% (0/400)** |

> Evaluated on full 400-sample held-out test set using GGUF Q4_K_M on CPU (llama-cpp-python).

**Key insight:** SFT teaches the model *what* JSON to produce. DPO teaches it to *prefer* clean-format output over malformed alternatives (fences, preambles, trailing commas, single quotes). The combination achieves near-perfect schema compliance.

---

## Quickstart (local CPU inference)

```bash
# 1. Clone and setup
git clone https://github.com/Tahleels/json-extract-qlora-dpo.git
cd json-extract-qlora-dpo
python -m venv venv && venv\Scripts\activate
pip install llama-cpp-python pydantic pyyaml

# 2. Download GGUF from HF Hub
# Place in: artifacts/model-Q4_K_M.gguf

# 3. Run inference
python -m src.infer "Hi, I'm Alex Rivera, 29. Lead Engineer at TechCorp in Chicago. alex@techcorp.io"
```

---

## Reproduce Training

All training runs on **free Google Colab T4 GPU**. Open the notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tahleels/json-extract-qlora-dpo/blob/main/notebooks/colab_end_to_end.ipynb)

The notebook is self-contained — it clones this repo, generates data, trains, merges, and exports GGUF.

### Hyperparameters

**SFT (`configs/sft.yaml`):**
- LoRA rank: 16, alpha: 32, dropout: 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Epochs: 3, batch size: 4, grad accum: 4, lr: 2e-4, max_seq_length: 512

**DPO (`configs/dpo.yaml`):**
- Beta: 0.1, Epochs: 1, batch size: 2, grad accum: 4, lr: 5e-6

---

## Dataset

**Fully synthetic** — generated using `Faker` with 14 sentence templates:
- **SFT:** 3,400 train / 200 val / 400 test — each `{input_text, output_json}` pair
- **DPO:** 2,000 pairs — `{prompt, chosen (clean JSON), rejected (corrupted JSON)}`

Corruption types for rejected samples: markdown fences, chatty preambles, trailing commas, single quotes, missing fields.

---

## File Structure

```
json-extract-qlora-dpo/
├── configs/
│   ├── base.yaml          # schema, model id, data counts
│   ├── sft.yaml           # SFT hyperparameters
│   └── dpo.yaml           # DPO hyperparameters
├── notebooks/
│   └── colab_end_to_end.ipynb
├── src/
│   ├── schema.py          # PersonRecord Pydantic model + validators
│   ├── config.py          # YAML config loader
│   ├── infer.py           # local CPU inference (single input)
│   ├── data/              # generate, build_sft, build_dpo, corrupt, validate
│   ├── train/             # train_sft, train_dpo, merge, export_gguf
│   ├── eval/              # evaluate_gguf (CPU benchmark on test set)
│   └── deploy/            # push_to_hub, app.py (Gradio Space)
└── data/samples/          # committed sample rows (full data gitignored)
```

---

## Cost

| Resource | Cost |
|---|---|
| Training (T4, ~40 min) | $0 (free Colab) |
| Hosting (HF Hub + Spaces) | $0 (free tier) |
| Inference (CPU laptop) | $0 |
| **Total** | **$0** |

---

*Built as part of an LLM fine-tuning portfolio project demonstrating QLoRA, DPO, quantization, and deployment on resource-constrained hardware.*
