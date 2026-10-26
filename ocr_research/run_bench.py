#!/usr/bin/env python3
"""Compare line OCR engines on the fixed Russian page sample.

Engines:
  tesseract     rus+eng, the current djvu_to_text.py settings
  rapid_eslav   RapidOCR PP-OCRv5, East Slavic recognizer
  rapid_cyr     RapidOCR PP-OCRv5, broader Cyrillic recognizer
  rapid_eslav_server  same recognizer, heavier PP-OCRv5 server detector

Column-aware reading order is applied on top of rapid_eslav and saved
as rapid_eslav_cols. That is layout, not a different model.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein
from rapidocr import RapidOCR
from rapidocr.utils.typings import ModelType, OCRVersion

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
SAMPLE = HERE / "sample"
PAGES = SAMPLE / "pages"
OUT = HERE / "results" / "text"
TESSERACT = ROOT / ".tools" / "tesseract" / "usr" / "bin" / "tesseract"
TESS_LIB = ROOT / ".tools" / "tesseract" / "usr" / "lib" / "x86_64-linux-gnu"
TESSDATA = ROOT / ".tools" / "tesseract" / "usr" / "share" / "tesseract-ocr" / "5" / "tessdata"
MODEL_DIR = HERE / ".cache" / "rapidocr"

WORD4 = re.compile(r"[а-я]{4,}")
WORD6 = re.compile(r"[а-я]{6,}")


def manifest() -> list[dict]:
    return json.loads((SAMPLE / "manifest.json").read_text(encoding="utf-8"))


def rapid_params(lang: str, det_model: str) -> dict:
    det = ModelType.SERVER if det_model == "server" else ModelType.MOBILE
    return {
        "Global.model_root_dir": str(MODEL_DIR),
        "Global.max_side_len": 2560,
        "Global.log_level": "warning",
        "EngineConfig.onnxruntime.intra_op_num_threads": 8,
        "Det.lang_type": "multi",
        "Det.model_type": det,
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "Cls.lang_type": "multi",
        "Cls.model_type": ModelType.MOBILE,
        "Cls.ocr_version": OCRVersion.PPOCRV5,
        "Rec.lang_type": lang,
        "Rec.model_type": ModelType.MOBILE,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
    }


def boxes_to_items(boxes, txts) -> list[dict]:
    items = []
    for box, text in zip(boxes, txts):
        arr = np.asarray(box, dtype=float)
        items.append(
            {
                "x0": float(arr[:, 0].min()),
                "y0": float(arr[:, 1].min()),
                "x1": float(arr[:, 0].max()),
                "y1": float(arr[:, 1].max()),
                "text": str(text).strip(),
            }
        )
    return [item for item in items if item["text"]]


def lines_of(items: list[dict]) -> str:
    if not items:
        return ""
    items = sorted(items, key=lambda item: (item["y0"], item["x0"]))
    lines: list[list[dict]] = []
    for item in items:
        if not lines:
            lines.append([item])
            continue
        prev = lines[-1][0]
        prev_h = max(8.0, prev["y1"] - prev["y0"])
        if abs(item["y0"] - prev["y0"]) < prev_h * 0.6:
            lines[-1].append(item)
        else:
            lines.append([item])
    out = []
    for line in lines:
        line.sort(key=lambda item: item["x0"])
        out.append(" ".join(item["text"] for item in line))
    return "\n".join(out)


def column_text(boxes, txts) -> str:
    """Read left column then right when a clear vertical gutter exists."""
    items = boxes_to_items(boxes, txts)
    if len(items) < 12:
        return lines_of(items)
    left_edge = min(item["x0"] for item in items)
    right_edge = max(item["x1"] for item in items)
    width = right_edge - left_edge
    if width < 100:
        return lines_of(items)
    mid = left_edge + width / 2
    band = width * 0.06
    centers = [(item["x0"] + item["x1"]) / 2 for item in items]
    gutter_hits = sum(1 for center in centers if abs(center - mid) < band)
    left = [item for item in items if (item["x0"] + item["x1"]) / 2 < mid]
    right = [item for item in items if (item["x0"] + item["x1"]) / 2 >= mid]
    if min(len(left), len(right)) < 8 or gutter_hits / len(items) > 0.12:
        return lines_of(items)
    return lines_of(left).rstrip() + "\n\n" + lines_of(right).rstrip() + "\n"


def tesseract(png: Path) -> str:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{TESS_LIB}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"
    env["TESSDATA_PREFIX"] = str(TESSDATA)
    env["OMP_THREAD_LIMIT"] = "1"
    result = subprocess.run(
        [
            str(TESSERACT),
            str(png),
            "stdout",
            "-l",
            "rus+eng",
            "--psm",
            "3",
            "--oem",
            "1",
        ],
        capture_output=True,
        timeout=180,
        check=False,
        env=env,
    )
    text = result.stdout.decode("utf-8", "replace")
    if result.returncode != 0 and not text.strip():
        err = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(err or f"tesseract exit {result.returncode}")
    return text


def run_rapid(name: str, lang: str, det_model: str, reorder: bool) -> None:
    engine = RapidOCR(params=rapid_params(lang, det_model))
    for item in manifest():
        dest = OUT / name / f"{item['id']}.txt"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"skip {name} {item['id']}", flush=True)
            continue
        started = time.perf_counter()
        result = engine(str(PAGES / f"{item['id']}.png"))
        if reorder and result.boxes is not None and result.txts is not None:
            text = column_text(result.boxes, result.txts)
        elif result.txts:
            text = result.to_markdown()
        else:
            text = ""
        elapsed = time.perf_counter() - started
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        print(
            f"{name:20} {item['id']:14} chars={len(text):5} sec={elapsed:6.1f}",
            flush=True,
        )


def run_tesseract() -> None:
    for item in manifest():
        dest = OUT / "tesseract" / f"{item['id']}.txt"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"skip tesseract {item['id']}", flush=True)
            continue
        started = time.perf_counter()
        text = tesseract(PAGES / f"{item['id']}.png")
        elapsed = time.perf_counter() - started
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        print(
            f"{'tesseract':20} {item['id']:14} chars={len(text):5} sec={elapsed:6.1f}",
            flush=True,
        )


def normalize(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = text.replace("ё", "е").replace("Ё", "Е").lower()
    text = re.sub(r"[#>*_`|]+", " ", text)
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def recall(pattern: re.Pattern[str], gt: str, hyp: str) -> float | None:
    words = pattern.findall(normalize(gt))
    if not words:
        return None
    found = set(pattern.findall(normalize(hyp)))
    return sum(1 for word in words if word in found) / len(words)


def cer(gt: str, hyp: str) -> float | None:
    a = normalize(gt)
    b = normalize(hyp)
    if not a:
        return None
    return Levenshtein.distance(a, b) / len(a)


def score() -> None:
    rows = []
    engines = sorted(path.name for path in OUT.iterdir() if path.is_dir())
    for item in manifest():
        gt = (PAGES / f"{item['id']}.gt.txt").read_text(encoding="utf-8")
        for engine in engines:
            hyp_path = OUT / engine / f"{item['id']}.txt"
            if not hyp_path.exists():
                continue
            hyp = hyp_path.read_text(encoding="utf-8")
            rows.append(
                {
                    "engine": engine,
                    "page": item["id"],
                    "has_gt": item["has_gt"],
                    "chars": len(hyp),
                    "recall4": recall(WORD4, gt, hyp),
                    "recall6": recall(WORD6, gt, hyp),
                    "cer": cer(gt, hyp),
                }
            )
    (HERE / "results" / "scores.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"{'engine':22} {'page':14} {'recall6':8} {'recall4':8} {'cer':8} chars")
    for row in rows:
        def fmt(value: float | None) -> str:
            return "   -   " if value is None else f"{value:8.3f}"

        print(
            f"{row['engine']:22} {row['page']:14} {fmt(row['recall6'])} "
            f"{fmt(row['recall4'])} {fmt(row['cer'])} {row['chars']}"
        )


def main() -> None:
    import sys

    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "line"
    if cmd == "line":
        run_tesseract()
        run_rapid("rapid_eslav", "eslav", "mobile", reorder=False)
        run_rapid("rapid_cyr", "cyrillic", "mobile", reorder=False)
        run_rapid("rapid_eslav_server", "eslav", "server", reorder=False)
        run_rapid("rapid_eslav_cols", "eslav", "mobile", reorder=True)
        score()
    elif cmd == "score":
        score()
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
