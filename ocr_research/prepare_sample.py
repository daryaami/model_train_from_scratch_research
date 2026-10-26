#!/usr/bin/env python3
"""Render a fixed set of Russian pages for the OCR comparison.

Ground truth, when it exists, is the DjVu text layer of that page.
Scans with an empty layer are kept so we can see the hard case.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "chem_dataset" / "chem_extracted"
DJVU_BIN = ROOT / ".tools" / "djvulibre" / "usr" / "bin"
SAMPLE = Path(__file__).resolve().parent / "sample"
PAGES_DIR = SAMPLE / "pages"

# Middle pages of ordinary textbooks, plus two scans Tesseract already struggles with.
PAGES = [
    {
        "id": "general",
        "file": "Мингулина Э.И.Курс общей химии.1990.djvu",
        "page": 153,
        "expect": "text_layer",
        "note": "учебник, сплошной текст",
    },
    {
        "id": "thermo",
        "file": "Крестов Г.А. - Термодинамика ионныx процессов (1984).djvu",
        "page": 46,
        "expect": "text_layer",
        "note": "формулы",
    },
    {
        "id": "ir",
        "file": "Смит А.Л. - Прикладная ИК-спектроскопия.. Основы, техника, аналитическое применение (1982).djvu",
        "page": 109,
        "expect": "text_layer",
        "note": "подписи к спектрам и формулы",
    },
    {
        "id": "colloid",
        "file": "Фролов Ю.Г. (ред) - Лабораторные работы и задачи в коллоидной химии (1986).djvu",
        "page": 71,
        "expect": "text_layer",
        "note": "лабораторный практикум",
    },
    {
        "id": "pharmacopeia",
        "file": "Государственная Фармакопея СССР. вып.2 (1990).djvu",
        "page": 65,
        "expect": "scan",
        "note": "скан без текстового слоя",
    },
    {
        "id": "allyl",
        "file": "Аллилирование фенолов и эфиров фенолов. I. Аллилирование гваякола в присутствии меди.djv",
        "page": 2,
        "expect": "scan",
        "note": "журнальная страница, текущий tesseract ее ломает",
    },
]

CYR = re.compile(r"[А-Яа-яЁё]")


def run(cmd: list[str], timeout: int = 60) -> subprocess.CompletedProcess[bytes]:
    env = os.environ.copy()
    env["PATH"] = f"{DJVU_BIN}{os.pathsep}{env.get('PATH', '')}"
    return subprocess.run(cmd, capture_output=True, timeout=timeout, check=False, env=env)


def main() -> None:
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for item in PAGES:
        src = SRC / item["file"]
        if not src.is_file():
            raise SystemExit(f"missing {src}")
        page = int(item["page"])
        gt = run([str(DJVU_BIN / "djvutxt"), f"-page={page}", str(src)])
        text = gt.stdout.decode("utf-8", "replace")
        letters = CYR.findall(text)
        alpha = sum(ch.isalpha() for ch in text)
        ppm = PAGES_DIR / f"{item['id']}.ppm"
        rendered = run(
            [
                str(DJVU_BIN / "ddjvu"),
                "-format=ppm",
                f"-page={page}",
                "-size=2200x3200",
                str(src),
                str(ppm),
            ],
            timeout=90,
        )
        if rendered.returncode != 0 or not ppm.exists():
            err = rendered.stderr.decode("utf-8", "replace")
            raise SystemExit(f"render failed {item['id']}: {err}")
        image = Image.open(ppm)
        png = PAGES_DIR / f"{item['id']}.png"
        pdf = PAGES_DIR / f"{item['id']}.pdf"
        image.save(png)
        image.save(pdf, "PDF", resolution=150.0)
        ppm.unlink()
        gt_path = PAGES_DIR / f"{item['id']}.gt.txt"
        gt_path.write_text(text, encoding="utf-8")
        record = {
            **item,
            "source": str(src.relative_to(ROOT)),
            "png": str(png.relative_to(SAMPLE)),
            "pdf": str(pdf.relative_to(SAMPLE)),
            "gt": str(gt_path.relative_to(SAMPLE)),
            "width": image.size[0],
            "height": image.size[1],
            "gt_chars": len(text.strip()),
            "gt_cyr_letters": len(letters),
            "gt_cyr_ratio": round(len(letters) / alpha, 3) if alpha else 0.0,
            "has_gt": len(letters) >= 400,
        }
        manifest.append(record)
        print(
            f"{item['id']:14} {image.size[0]}x{image.size[1]} "
            f"gt_chars={record['gt_chars']} cyr={record['gt_cyr_ratio']} "
            f"has_gt={record['has_gt']}"
        )
    (SAMPLE / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {SAMPLE / 'manifest.json'}")


if __name__ == "__main__":
    main()
