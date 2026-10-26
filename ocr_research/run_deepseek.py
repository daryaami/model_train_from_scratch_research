#!/usr/bin/env python3
"""DeepSeek-OCR-2 on the fixed sample pages.

Grounding prompt, eager attention (no flash-attn). The bundled infer()
calls .cuda(), so the weights stay on the one visible GPU.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / ".cache" / "hf"))

import torch

# The model code targets transformers 4.46. Flash attention is unused
# when attn_implementation is eager, but the import still names the class.
import transformers.models.llama.modeling_llama as _llama_mod

if not hasattr(_llama_mod, "LlamaFlashAttention2"):
    _llama_mod.LlamaFlashAttention2 = _llama_mod.LlamaAttention

from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
PAGES = HERE / "sample" / "pages"
OUT = HERE / "results" / "deepseek"
MODEL = "deepseek-ai/DeepSeek-OCR-2"
PROMPT = "<image>\n<|grounding|>Convert the document to markdown. "
REF = re.compile(
    r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>",
    flags=re.DOTALL,
)
PAGES_TO_RUN = ["general", "thermo", "ir", "colloid", "pharmacopeia", "allyl"]


def clean(text: str) -> str:
    text = REF.sub("", text)
    text = text.replace("<｜end▁of▁sentence｜>", "")
    return text.strip()


def main() -> None:
    pages = sys.argv[1:] or PAGES_TO_RUN
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model = model.eval().cuda()
    for page in pages:
        dest = OUT / f"{page}.md"
        image = PAGES / f"{page}.png"
        print(f"START {page} {time.strftime('%H:%M:%S')}", flush=True)
        started = time.perf_counter()
        raw = model.infer(
            tokenizer,
            prompt=PROMPT,
            image_file=str(image),
            output_path=str(OUT / page),
            base_size=1024,
            image_size=768,
            crop_mode=True,
            save_results=False,
            eval_mode=True,
        )
        text = clean(raw or "")
        dest.write_text(text, encoding="utf-8")
        elapsed = time.perf_counter() - started
        cyr = sum(1 for ch in text.lower() if "а" <= ch <= "я")
        print(f"DONE {page} sec={elapsed:.1f} chars={len(text)} cyr={cyr}", flush=True)


if __name__ == "__main__":
    main()
