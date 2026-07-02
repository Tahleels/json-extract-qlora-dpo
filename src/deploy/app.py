"""
Gradio Space demo for JSON Extraction model (GGUF, CPU-friendly).
Deploy this file to Hugging Face Spaces (SDK: gradio, hardware: CPU basic).

The Space will automatically download the GGUF from your HF model repo.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# ── Config ────────────────────────────────────────────────────────────────────
HF_REPO_ID  = "Tahleels/qwen2.5-0.5b-json-extraction-qlora-dpo"
GGUF_FILE   = "gguf/model-Q4_K_M.gguf"
LOCAL_PATH  = "/tmp/model-Q4_K_M.gguf"

SYSTEM_PROMPT = (
    "You are a JSON extraction engine. "
    "Given unstructured text, output ONLY a single minified JSON object "
    "with these exact keys: name, age, job_title, company, city, email. "
    "No markdown, no explanation, no extra text. "
    "Output ONLY valid JSON."
)

SCHEMA_KEYS = ["name", "age", "job_title", "company", "city", "email"]

EXAMPLES = [
    "Hi, I'm Sarah Chen, 34. I work as a Senior Data Analyst at TechCorp in Seattle. Reach me at sarah.chen@techcorp.io.",
    "My name's James Miller, 28, software engineer at Stripe. Based in San Francisco. james.miller@stripe.com.",
    "Priya Sharma here — 31, product manager at Flipkart in Bangalore. You can email me at p.sharma@flipkart.in.",
]

# ── Load model (once, at startup) ─────────────────────────────────────────────
print("Downloading GGUF model...")
if not Path(LOCAL_PATH).exists():
    hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=GGUF_FILE,
        local_dir="/tmp",
        local_dir_use_symlinks=False,
    )
    import shutil
    shutil.move(f"/tmp/{GGUF_FILE.split('/')[-1]}", LOCAL_PATH)

print("Loading model into llama.cpp...")
llm = Llama(
    model_path=LOCAL_PATH,
    n_ctx=512,
    n_threads=int(os.environ.get("LLAMA_THREADS", "2")),
    verbose=False,
)
print("Model loaded ✅")

# ── Inference function ────────────────────────────────────────────────────────
def extract_json(input_text: str) -> tuple[str, str]:
    """Returns (raw_output, validation_status)."""
    if not input_text.strip():
        return "", "⚠️ Please enter some text."

    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{input_text.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    output = llm(
        prompt,
        max_tokens=256,
        stop=["<|im_end|>", "<|endoftext|>"],
        temperature=0.0,
    )
    raw = output["choices"][0]["text"].strip()

    # Validate
    try:
        obj = json.loads(raw)
        missing = [k for k in SCHEMA_KEYS if k not in obj]
        extra   = [k for k in obj if k not in SCHEMA_KEYS]

        if not missing and not extra:
            pretty = json.dumps(obj, indent=2)
            status = "✅ Valid — schema matches perfectly"
        elif missing:
            pretty = json.dumps(obj, indent=2)
            status = f"⚠️ Parsed but missing fields: {missing}"
        else:
            pretty = json.dumps(obj, indent=2)
            status = f"⚠️ Parsed but extra fields: {extra}"
    except json.JSONDecodeError:
        pretty = raw
        status = "❌ Not valid JSON — model output could not be parsed"

    return pretty, status


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="JSON Extraction — QLoRA + DPO Fine-tuned", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# 🧠 JSON Extraction — QLoRA + DPO Fine-tuned
**Model:** Qwen2.5-0.5B-Instruct fine-tuned with QLoRA SFT + DPO  
**Task:** Convert unstructured text → strict JSON schema  
**Schema:** `name`, `age`, `job_title`, `company`, `city`, `email`

> Fine-tuned on free Colab T4 GPU. Runs on CPU via GGUF Q4_K_M quantization.
    """)

    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(
                label="📝 Input Text",
                placeholder="e.g. Hi, I'm Alex Rivera, 29. I work as Lead Engineer at TechCorp in Chicago...",
                lines=4,
            )
            run_btn = gr.Button("Extract JSON →", variant="primary")
            gr.Examples(examples=EXAMPLES, inputs=input_box)

        with gr.Column():
            output_box = gr.Code(
                label="📤 Extracted JSON",
                language="json",
                lines=10,
            )
            status_box = gr.Textbox(label="✔ Validation", interactive=False)

    run_btn.click(
        fn=extract_json,
        inputs=[input_box],
        outputs=[output_box, status_box],
    )
    input_box.submit(
        fn=extract_json,
        inputs=[input_box],
        outputs=[output_box, status_box],
    )

    gr.Markdown("""
---
**GitHub:** [Tahleels/json-extract-qlora-dpo](https://github.com/Tahleels/json-extract-qlora-dpo) · 
**Model:** [Tahleels/qwen2.5-0.5b-json-extraction-qlora-dpo](https://huggingface.co/Tahleels/qwen2.5-0.5b-json-extraction-qlora-dpo)
    """)

if __name__ == "__main__":
    demo.launch()
