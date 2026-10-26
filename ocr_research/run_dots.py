#!/usr/bin/env python3
"""dots.ocr on the fixed sample pages.

Layout prompt: formulas as LaTeX, tables as HTML, original text, reading order.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / ".cache" / "hf"))

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoProcessor

HERE = Path(__file__).resolve().parent
PAGES = HERE / "sample" / "pages"
OUT = HERE / "results" / "dots"
MODEL = str(HERE / ".cache" / "dots_ocr_local")
PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""
PAGES_TO_RUN = ["general", "thermo", "ir", "colloid", "pharmacopeia", "allyl"]


def parse_layout(raw: str):
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start_candidates = [i for i in (text.find("["), text.find("{")) if i >= 0]
    if not start_candidates:
        return None
    blob = text[min(start_candidates) :]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        end = max(blob.rfind("]"), blob.rfind("}"))
        if end < 0:
            return None
        try:
            return json.loads(blob[: end + 1])
        except json.JSONDecodeError:
            return None


def elements(data) -> list:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("elements", "layout", "data", "result"):
            if isinstance(data.get(key), list):
                return data[key]
        if "category" in data or "text" in data:
            return [data]
    return []


def stitch(data) -> str:
    parts = []
    for el in elements(data):
        if not isinstance(el, dict):
            continue
        cat = str(el.get("category") or el.get("type") or "")
        text = str(el.get("text") or "").strip()
        if cat == "Picture" or not text:
            continue
        if cat == "Formula":
            parts.append(f"$$\n{text}\n$$")
        elif cat in {"Title", "Section-header"}:
            parts.append(f"## {text}")
        else:
            parts.append(text)
    return "\n\n".join(parts)


def main() -> None:
    pages = sys.argv[1:] or PAGES_TO_RUN
    OUT.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model = model.eval().cuda()
    for page in pages:
        image = PAGES / f"{page}.png"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image)},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        print(f"START {page} {time.strftime('%H:%M:%S')}", flush=True)
        started = time.perf_counter()
        templated = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[templated],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=8192, do_sample=False)
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        raw = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        (OUT / f"{page}.raw.txt").write_text(raw, encoding="utf-8")
        data = parse_layout(raw)
        text = stitch(data) if data is not None else raw.strip()
        (OUT / f"{page}.md").write_text(text, encoding="utf-8")
        elapsed = time.perf_counter() - started
        cyr = sum(1 for ch in text.lower() if "а" <= ch <= "я")
        print(
            f"DONE {page} sec={elapsed:.1f} chars={len(text)} cyr={cyr} json={data is not None}",
            flush=True,
        )


if __name__ == "__main__":
    main()
