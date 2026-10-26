#!/usr/bin/env python3
"""Docling on the sample pages: layout, tables, and formula-to-LaTeX.

Text OCR is Tesseract rus+eng, which won the line-level comparison.
Formula regions are passed through Docling's formula model.
CPU only, so this does not touch the GPU used by training.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TESSERACT = ROOT / ".tools" / "tesseract" / "usr" / "bin" / "tesseract"
TESS_LIB = ROOT / ".tools" / "tesseract" / "usr" / "lib" / "x86_64-linux-gnu"
TESSDATA = ROOT / ".tools" / "tesseract" / "usr" / "share" / "tesseract-ocr" / "5" / "tessdata"

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["DOCLING_DEVICE"] = "cpu"
os.environ["HF_HOME"] = str(HERE / ".cache" / "hf")
os.environ["TESSDATA_PREFIX"] = str(TESSDATA)
os.environ["LD_LIBRARY_PATH"] = (
    f"{TESS_LIB}{os.pathsep}{os.environ.get('LD_LIBRARY_PATH', '')}"
)

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    OcrMode,
    PdfPipelineOptions,
    TableFormerMode,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

PAGES = ["general", "thermo", "allyl", "pharmacopeia", "ir", "colloid"]
OUT = HERE / "results" / "text" / "docling"


def build() -> DocumentConverter:
    ocr = TesseractCliOcrOptions(
        lang=["rus", "eng"],
        tesseract_cmd=str(TESSERACT),
        path=str(TESSDATA),
        psm=3,
        mode=OcrMode.FULL_PAGE,
        scale=2.0,
    )
    pipeline = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        do_formula_enrichment=True,
        generate_page_images=False,
        images_scale=2.0,
        ocr_options=ocr,
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CPU,
            num_threads=8,
        ),
    )
    pipeline.table_structure_options.mode = TableFormerMode.ACCURATE
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline),
        }
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    converter = build()
    for page_id in PAGES:
        dest = OUT / f"{page_id}.md"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"skip {page_id}", flush=True)
            continue
        src = HERE / "sample" / "pages" / f"{page_id}.pdf"
        started = time.perf_counter()
        result = converter.convert(str(src))
        text = result.document.export_to_markdown()
        dest.write_text(text, encoding="utf-8")
        print(
            f"docling {page_id:14} chars={len(text):5} sec={time.perf_counter() - started:6.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
