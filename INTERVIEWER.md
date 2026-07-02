# Project Deep-Dive — For Interviews

> This document explains every design decision made in this project in plain English.
> Use this to prepare for "walk me through your project" questions.

---

## The One-Line Summary

> Fine-tuned a small open-source LLM to reliably convert messy natural language text into a strict, validated JSON object — with 100% parse rate and zero formatting errors — running entirely on free hardware.

---

## 1. What problem does this solve?

### The real-world use case
Imagine you have thousands of text blobs — LinkedIn bios, email signatures, CRM notes — and you need to extract structured data (name, company, email) from all of them. You could use regex, but natural language doesn't follow patterns. You could use a big GPT-4 API, but that costs money per call and leaks your data to a third party.

The answer: fine-tune a **small, local model** to do exactly this one job perfectly.

### Why is this hard without fine-tuning?
The base Qwen2.5-0.5B model, out of the box, only produces valid JSON about **62% of the time**. The other 38% it adds markdown fences like:
```
```json
{"name": "Sarah"}
```
```
...or a chatty preamble like "Sure! Here is the extracted JSON: {...}", or it misses fields entirely. This is unusable in a production pipeline.

---

## 2. Why Qwen2.5-0.5B?

### The decision
We deliberately picked the **smallest capable model** instead of a large one.

### Why not a bigger model (7B, 13B)?
| Reason | Explanation |
|---|---|
| Free Colab T4 only has 16GB VRAM | A 7B model in 4-bit quantization alone takes ~4GB. After LoRA, optimizer states, and activations there's no room |
| The task doesn't need world knowledge | Extracting 6 fields from 1 sentence is a *format* problem, not a *knowledge* problem |
| Deployment target is a CPU laptop | A 0.5B GGUF is ~400MB. A 7B GGUF is ~4GB. Users can't run that locally |
| Training time | 0.5B trains SFT in ~15 min on a free T4. 7B would take 2-3 hours and likely OOM |

### Why not an even smaller model (0.1B)?
At sub-0.5B, models start to struggle to follow multi-field instruction templates reliably. 0.5B is the sweet spot for this task.

### Why Qwen over Mistral/Llama?
- Qwen2.5-0.5B has an excellent **chat template** (`<|im_start|>system/user/assistant`) which makes instruction-following much more controllable
- At 0.5B, Qwen consistently outperforms Mistral and Llama variants of similar size on structured output tasks
- Apache 2.0 license — no restrictions on commercial use

---

## 3. Why QLoRA (4-bit + LoRA)?

### What is QLoRA?
QLoRA = Quantized base model + LoRA adapters. Instead of fine-tuning all 500M parameters:
1. Load the base model in **4-bit NF4 quantization** (saves ~75% memory vs float32)
2. Freeze all original weights
3. Add small **LoRA adapter layers** (rank-16 matrices) to only the attention and MLP projections
4. Train only those adapter layers (< 1% of total parameters)

### Why not full fine-tuning?
Full fine-tuning of 500M parameters would require at minimum 4GB of VRAM just for gradients, plus activations. On a free T4 this would OOM. QLoRA makes training feasible on any 16GB GPU.

### Why not just prompt engineering?
Prompt engineering alone can get the base model to ~80-85% valid JSON. But:
- It still fails on unusual sentence structures
- You can't eliminate the markdown fence problem without training
- Consistent 100% compliance requires the model to have truly *learned* the behavior, not just been reminded of it

### LoRA Hyperparameter Choices
| Parameter | Value | Reasoning |
|---|---|---|
| Rank (r) | 16 | Enough capacity to learn format behavior without overfitting |
| Alpha | 32 | Standard 2x rank — controls effective learning rate of the adapter |
| Dropout | 0.05 | Light regularization on a small dataset |
| Target modules | All projection layers (`q/k/v/o/gate/up/down_proj`) | JSON format is a global behavior — you want all layers to participate |

---

## 4. Why DPO after SFT?

### What SFT alone gives you
After SFT, the model outputs valid JSON most of the time. But SFT is trained on only the *correct* output. The model doesn't know what bad output looks like, so it can still occasionally produce a markdown fence or a preamble.

### What DPO adds
DPO (Direct Preference Optimization) is trained on **pairs**:
- `chosen`: the clean, correct JSON output
- `rejected`: a deliberately corrupted version (fences, preambles, trailing commas, single quotes, missing fields)

DPO explicitly teaches the model: *"when you see this input, prefer this output format over that one."* It uses the SFT model as a reference and pushes the output distribution away from rejected formats.

### Why DPO over RLHF?
RLHF requires training a separate reward model and running PPO — which is extremely unstable and memory-intensive. DPO achieves the same preference-learning result with a simpler, more stable training loop. It requires no reward model and works well even on small models.

### Results of combining SFT + DPO
| Stage | Valid JSON % | Schema Compliance % |
|---|---|---|
| Base model | ~62% | ~40% |
| After SFT | ~91% | ~88% |
| After DPO | **100%** | **100%** |

---

## 5. Why synthetic data?

### The decision
All 5,400+ training examples were generated programmatically using Python's `Faker` library.

### Why not scrape real data?
| Issue | Detail |
|---|---|
| Privacy | Real person records (name, email, company) are PII — you can't train on scraped LinkedIn data legally |
| Labelling cost | Getting 5,000 real text blobs hand-annotated would take weeks and money |
| Coverage | Real data has uneven distribution — synthetic data lets you guarantee coverage of all 14 sentence templates |

### The risk: distribution mismatch
The model is trained and tested on the same synthetic distribution. In production, real text will look different. This is acknowledged in the project — it's a portfolio project demonstrating the *technique*, not a production system. The next step would be collecting a small real-world evaluation set.

---

## 6. Why GGUF quantization?

### What is GGUF?
GGUF is a file format developed by `llama.cpp` for storing quantized LLMs. After training, we merge the LoRA adapter into the base model and convert to GGUF Q4_K_M format.

### What does Q4_K_M mean?
- **Q4**: 4-bit quantization (each weight stored in 4 bits instead of 16 bits = 75% size reduction)
- **K_M**: "K-quant medium" — a smarter quantization scheme that allocates more bits to important layers and fewer to less critical layers. Gives better quality vs a naive uniform 4-bit scheme

### Why GGUF over safetensors?
| Format | Inference Library | CPU Friendly? | Size |
|---|---|---|---|
| safetensors | transformers, PyTorch | No (slow) | ~1GB (fp16) |
| GGUF Q4_K_M | llama.cpp | Yes (optimized) | ~400MB |

`llama-cpp-python` uses highly optimized SIMD instructions for CPU matrix multiplication, making CPU inference practical for a 0.5B model.

---

## 7. Why this tech stack?

| Tool | Role | Why this over alternatives |
|---|---|---|
| **TRL (SFTTrainer, DPOTrainer)** | Training loop | Purpose-built for RLHF/DPO workflows. Handles chat templates, gradient accumulation, and LoRA integration natively |
| **PEFT** | LoRA adapter management | The standard library for parameter-efficient fine-tuning in the HuggingFace ecosystem |
| **bitsandbytes** | 4-bit NF4 quantization | The only library that integrates 4-bit training directly into PyTorch |
| **Faker** | Synthetic data generation | Locale-aware, produces realistic names/emails/companies. Much better than hardcoding |
| **Pydantic** | Schema validation | Strict type enforcement at both data generation and evaluation time |
| **llama-cpp-python** | CPU inference | Python bindings for llama.cpp — the gold standard for fast CPU LLM inference |
| **Gradio** | Demo UI | 10-line setup for a hosted interactive ML demo |

---

## 8. Architecture Decision: Why local eval, not cloud?

We run the evaluation harness **locally on CPU** (not in Colab) for a specific reason: the GGUF model is the artifact you'd actually deploy in the real world. Evaluating the GGUF specifically (not the fp16 PyTorch weights) tells you the true performance of what you're shipping.

It also validates the complete pipeline: `training → merge → quantize → GGUF → CPU inference → eval`.

---

## 9. What I would do differently in production

1. **Real-world eval set** — 200 manually labeled real text examples to measure distribution shift
2. **Streaming output** — Stream tokens in the Gradio UI instead of waiting for the full generation
3. **JSON repair fallback** — If the model outputs malformed JSON, attempt auto-repair before failing
4. **Confidence scoring** — Add a secondary classifier to flag low-confidence extractions for human review
5. **Larger base model** — With a proper A100 GPU, use Qwen2.5-7B for much higher accuracy on edge cases

---

## 10. Common interview questions — answered

**Q: Why is your evaluation accuracy 100%? Isn't that suspicious?**
> The task is narrow and well-defined. The model is evaluated on the same synthetic distribution it was trained on, so high numbers are expected. The interesting result is the *comparison* — base model at 62% vs fine-tuned at 100% — which quantifies the value of the training pipeline.

**Q: How would this scale to millions of documents?**
> You'd replace the single GGUF inference with a batched vLLM or TensorRT-LLM endpoint serving the fp16 model on a GPU. The GGUF is for local/demo use only.

**Q: What's the difference between SFT and DPO in simple terms?**
> SFT is teaching by example: "when you see X, output Y." DPO is teaching by contrast: "when you see X, prefer Y over Z." DPO is what eliminates the last 9% of formatting failures.

**Q: Why not just use GPT-4 with a JSON mode prompt?**
> GPT-4 JSON mode works well but costs ~$0.01 per call. At 1 million documents, that's $10,000. This fine-tuned 0.5B model runs at $0 per call on a CPU laptop, permanently, with no data sent to a third party.
