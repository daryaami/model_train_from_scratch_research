#!/usr/bin/env python3
"""Convert all DjVu books to UTF-8 text.

Phase 1 extracts a hidden text layer when it exists.
Phase 2 renders remaining scans and runs Tesseract (rus+eng).
Safe to rerun: finished .txt files are skipped.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "chem_dataset" / "chem_extracted"
OUT = ROOT / "data" / "chem_dataset" / "chem_text"
LOG_DIR = ROOT / "logs"
MANIFEST = OUT / "_manifest.jsonl"

DJVU_BIN = ROOT / ".tools" / "djvulibre" / "usr" / "bin"
TESS_ROOT = ROOT / ".tools" / "tesseract"
TESSERACT = TESS_ROOT / "usr" / "bin" / "tesseract"
TESS_LIB = TESS_ROOT / "usr" / "lib" / "x86_64-linux-gnu"
TESSDATA = TESS_ROOT / "usr" / "share" / "tesseract-ocr" / "5" / "tessdata"

OCR_WORKERS = int(os.environ.get("DJVU_OCR_WORKERS", "12"))
LAYER_WORKERS = int(os.environ.get("DJVU_LAYER_WORKERS", "16"))
PAGE_TIMEOUT = int(os.environ.get("DJVU_OCR_PAGE_TIMEOUT", "60"))
TEXT_LAYER_MIN_CHARS = 200
EXTENSIONS = {".djvu", ".djv"}


def configure_env() -> None:
    os.environ["PATH"] = f"{DJVU_BIN}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = (
        f"{TESS_LIB}{os.pathsep}{os.environ.get('LD_LIBRARY_PATH', '')}"
    )
    os.environ["TESSDATA_PREFIX"] = str(TESSDATA)
    os.environ["OMP_THREAD_LIMIT"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"


def run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)


def page_count(path: Path) -> int:
    result = run(["djvused", str(path), "-e", "n"], timeout=30)
    if result.returncode == 0:
        try:
            return int(result.stdout.decode("utf-8", "replace").strip())
        except ValueError:
            pass
    dump = run(["djvudump", str(path)], timeout=60)
    return dump.stdout.decode("utf-8", "replace").count("FORM:DJVU")


def extract_text_layer(path: Path) -> str:
    result = run(["djvutxt", str(path)], timeout=120)
    if result.returncode != 0:
        return ""
    return result.stdout.decode("utf-8", "replace")


def ocr_page(path: Path, page: int, workdir: Path) -> str:
    tiff = workdir / f"page_{page:05d}.tif"
    try:
        render = run(
            ["ddjvu", "-format=tiff", f"-page={page}", str(path), str(tiff)],
            timeout=PAGE_TIMEOUT,
        )
        if render.returncode != 0 or not tiff.exists() or tiff.stat().st_size == 0:
            err = render.stderr.decode("utf-8", "replace").strip()
            return f"[OCR_FAILED page {page}] render: {err}\n"
        ocr = run(
            [
                str(TESSERACT),
                str(tiff),
                "stdout",
                "-l",
                "rus+eng",
                "--psm",
                "3",
                "--oem",
                "1",
            ],
            timeout=PAGE_TIMEOUT,
        )
        text = ocr.stdout.decode("utf-8", "replace")
        if ocr.returncode != 0 and not text.strip():
            err = ocr.stderr.decode("utf-8", "replace").strip()
            return f"[OCR_FAILED page {page}] tesseract: {err}\n"
        return text
    finally:
        if tiff.exists():
            tiff.unlink()


def out_path_for(src: Path) -> Path:
    return OUT / src.relative_to(SRC).with_suffix(".txt")


def is_done(src: Path) -> bool:
    dest = out_path_for(src)
    return dest.exists() and dest.stat().st_size > 100


def write_manifest_record(record: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def convert_text_layer(src: Path) -> dict:
    configure_env()
    dest = out_path_for(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "source": str(src.relative_to(SRC)),
        "output": str(dest.relative_to(OUT)),
        "started": datetime.now(timezone.utc).isoformat(),
        "phase": "text_layer",
    }
    if is_done(src):
        record.update(status="skipped", reason="exists")
        record["finished"] = datetime.now(timezone.utc).isoformat()
        return record
    try:
        pages = page_count(src)
        record["pages"] = pages
        layer = extract_text_layer(src)
        if pages > 0 and len(layer.strip()) >= max(TEXT_LAYER_MIN_CHARS, pages * 20):
            dest.write_text(layer, encoding="utf-8")
            record.update(status="ok", method="text_layer", chars=len(layer))
        else:
            record.update(status="needs_ocr", chars=len(layer.strip()))
    except subprocess.TimeoutExpired:
        record.update(status="needs_ocr", error="timeout")
    except Exception as exc:  # noqa: BLE001
        record.update(status="needs_ocr", error=str(exc))
    record["finished"] = datetime.now(timezone.utc).isoformat()
    return record


def convert_ocr(src: Path) -> dict:
    configure_env()
    dest = out_path_for(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "source": str(src.relative_to(SRC)),
        "output": str(dest.relative_to(OUT)),
        "started": datetime.now(timezone.utc).isoformat(),
        "phase": "ocr",
    }
    if is_done(src):
        record.update(status="skipped", reason="exists")
        record["finished"] = datetime.now(timezone.utc).isoformat()
        return record
    partial = dest.with_suffix(".txt.partial")
    try:
        pages = page_count(src)
        record["pages"] = pages
        if pages <= 0:
            raise RuntimeError("could not determine page count")
        parts = [
            f"# source: {src.name}\n# pages: {pages}\n# method: tesseract rus+eng\n"
        ]
        with tempfile.TemporaryDirectory(prefix="djvu_ocr_") as tmp:
            workdir = Path(tmp)
            for page in range(1, pages + 1):
                text = ocr_page(src, page, workdir)
                parts.append(f"\n\n----- page {page} -----\n\n{text.rstrip()}\n")
                if page % 25 == 0 or page == pages:
                    partial.write_text("".join(parts), encoding="utf-8")
        text = "".join(parts)
        dest.write_text(text, encoding="utf-8")
        if partial.exists():
            partial.unlink()
        record.update(status="ok", method="tesseract", chars=len(text))
    except subprocess.TimeoutExpired:
        record.update(status="error", error="timeout")
    except Exception as exc:  # noqa: BLE001
        record.update(status="error", error=str(exc))
    record["finished"] = datetime.now(timezone.utc).isoformat()
    return record


def iter_djvu() -> list[Path]:
    files = [
        path
        for path in SRC.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    ]
    files.sort()
    return files


def run_pool(fn, files: list[Path], workers: int, label: str) -> list[dict]:
    if not files:
        print(f"{label}: nothing to do", flush=True)
        return []
    print(f"{label}: {len(files)} files, workers={workers}", flush=True)
    records = []
    ok = skipped = other = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=configure_env) as pool:
        futures = {pool.submit(fn, path): path for path in files}
        for i, future in enumerate(as_completed(futures), 1):
            src = futures[future]
            try:
                record = future.result()
            except Exception as exc:  # noqa: BLE001
                record = {
                    "source": str(src),
                    "status": "error",
                    "error": f"worker crashed: {exc}",
                    "phase": label,
                }
            write_manifest_record(record)
            records.append(record)
            status = record.get("status")
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                other += 1
            extra = ""
            if status == "ok":
                extra = (
                    f" {record.get('method')} pages={record.get('pages')} "
                    f"chars={record.get('chars')}"
                )
            elif status == "error":
                extra = f" {record.get('error')}"
            elif status == "needs_ocr":
                extra = f" layer_chars={record.get('chars', 0)}"
            print(f"[{label} {i}/{len(files)}] {status} {src.name[:90]}{extra}", flush=True)
    print(f"{label} done ok={ok} skipped={skipped} other={other}", flush=True)
    return records


def main() -> int:
    configure_env()
    if not SRC.is_dir():
        print(f"missing source dir: {SRC}", file=sys.stderr)
        return 1
    if not TESSERACT.is_file():
        print(f"missing tesseract: {TESSERACT}", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    files = iter_djvu()
    print(f"djvu files: {len(files)}", flush=True)

    layer_pending = [path for path in files if not is_done(path)]
    run_pool(convert_text_layer, layer_pending, LAYER_WORKERS, "layer")

    ocr_pending = [path for path in files if not is_done(path)]
    run_pool(convert_ocr, ocr_pending, OCR_WORKERS, "ocr")

    remaining = sum(1 for path in files if not is_done(path))
    print(f"finished remaining={remaining}", flush=True)
    return 0 if remaining == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
