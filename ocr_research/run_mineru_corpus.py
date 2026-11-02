#!/usr/bin/env python3
"""OCR the chemistry collection with MinerU VLM.

One book at a time. DjVu is rendered to a temporary PDF first.
Finished books are skipped, so the run can be started again after a stop.
Only books with Cyrillic are taken. The queue runs from the end, while
the cloud run walks from the start. The same manifest is used by both,
so a book finished by either path is not processed again.
Output is markdown in data/chem_dataset/chem_text_mineru.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mineru_ledger as ledger
from run_mineru_cloud import contains_cyrillic

ROOT = Path(__file__).resolve().parents[1]
SRC = ledger.SRC
OUT = ledger.OUT
DJVU_BIN = ROOT / ".tools" / "djvulibre" / "usr" / "bin"
DDJVU = DJVU_BIN / "ddjvu"
DJVUSED = DJVU_BIN / "djvused"
MINERU = ROOT / "ocr_research" / ".venv-mineru" / "bin" / "mineru-kit"
EXTENSIONS = {".djvu", ".djv", ".pdf"}
DATA_IMAGE = re.compile(r"!\[[^\]]*\]\(data:image/[^)]+\)")


def mineru_env() -> dict[str, str]:
    env = os.environ.copy()
    # Marker's CUDA llama is a different build. Leaving it on the path
    # makes mineru_llama_cpp fail to import.
    env.pop("LD_LIBRARY_PATH", None)
    env["PATH"] = f"{DJVU_BIN}{os.pathsep}{env.get('PATH', '')}"
    env["MINERU_MODEL_VLM_ENGINE"] = "llama-cpp"
    env["MINERU_MODEL_SMALL_BACKEND"] = "onnx"
    env["HF_HOME"] = str(ROOT / "ocr_research" / ".cache" / "hf")
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    return env


def run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=mineru_env(),
    )


def page_count(path: Path) -> int:
    if path.suffix.lower() == ".pdf":
        result = run(["pdfinfo", str(path)], timeout=120)
        for line in result.stdout.splitlines():
            if line.startswith("Pages:"):
                return int(line.split()[1])
        return 0
    result = run([str(DJVUSED), str(path), "-e", "n"], timeout=120)
    text = result.stdout.strip()
    return int(text) if text.isdigit() else 0


def sources() -> list[Path]:
    files = [
        path
        for path in SRC.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    ]
    files.sort()
    return files


def out_path_for(src: Path) -> Path:
    return ledger.canonical_output(src)


def strip_images(path: Path) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    cleaned, count = DATA_IMAGE.subn("![]()", text)
    if count:
        path.write_text(cleaned, encoding="utf-8")
    return count


def cloud_has_parts(src: Path) -> bool:
    parts = ledger.canonical_output(src).with_suffix(".parts")
    return parts.is_dir() and any(parts.glob("*.md"))


def convert_one(src: Path) -> dict:
    existing = ledger.finished_output(src)
    if existing is not None or cloud_has_parts(src):
        return {
            "source": ledger.source_key(src),
            "output": ledger.output_field(existing),
            "backend": "local",
            "status": "skipped",
            "started": ledger.now(),
            "finished": ledger.now(),
        }
    dest = out_path_for(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".md.partial")
    record = {
        "source": ledger.source_key(src),
        "output": ledger.output_field(dest),
        "backend": "local",
        "started": ledger.now(),
    }
    pages = page_count(src)
    record["pages"] = pages
    timeout = max(600, pages * 40 + 300)
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="mineru_corpus_") as tmp:
            work = Path(tmp)
            if src.suffix.lower() == ".pdf":
                pdf = src
            else:
                pdf = work / "pages.pdf"
                rendered = run(
                    [str(DDJVU), "-format=pdf", "-size=1700x2400", str(src), str(pdf)],
                    timeout=max(180, pages * 5 + 60),
                )
                if rendered.returncode != 0 or not pdf.exists():
                    raise RuntimeError(
                        f"ddjvu failed: {rendered.stderr[-500:] or rendered.stdout[-500:]}"
                    )
            if partial.exists():
                partial.unlink()
            parsed = run(
                [
                    str(MINERU),
                    "parse",
                    str(pdf),
                    "-o",
                    str(partial),
                    "--tier",
                    "standard",
                    "--ocr-mode",
                    "ocr",
                ],
                timeout=timeout,
            )
            log = (parsed.stdout + "\n" + parsed.stderr)[-2000:]
            if parsed.returncode != 0 or "Failed to parse" in log or not partial.is_file():
                raise RuntimeError(log.strip() or f"mineru exit {parsed.returncode}")
            if partial.stat().st_size < 20:
                raise RuntimeError("empty markdown")
            record["images_stripped"] = strip_images(partial)
            partial.replace(dest)
            record.update(
                status="ok",
                chars=dest.stat().st_size,
                seconds=round(time.monotonic() - started, 1),
            )
    except subprocess.TimeoutExpired:
        record.update(status="error", error="timeout", seconds=round(time.monotonic() - started, 1))
    except Exception as exc:  # noqa: BLE001
        record.update(status="error", error=str(exc)[:1000], seconds=round(time.monotonic() - started, 1))
    finally:
        if partial.exists() and record.get("status") != "ok":
            partial.unlink(missing_ok=True)
    record["finished"] = ledger.now()
    return record


def main() -> int:
    if not SRC.is_dir():
        print(f"missing source dir: {SRC}", file=sys.stderr)
        return 1
    if not MINERU.is_file():
        print(f"missing mineru: {MINERU}", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    files = sources()
    finished = ledger.completed_sources()
    pending = [path for path in files if ledger.source_key(path) not in finished]
    todo = []
    skipped = 0
    for path in pending:
        if contains_cyrillic(path):
            todo.append(path)
        else:
            skipped += 1
    # Cloud walks this list forward. Local walks it backward.
    todo.reverse()
    print(
        f"corpus files={len(files)} already_ok={len(finished)} cyrillic={len(todo)} skip_latin={skipped}",
        flush=True,
    )
    for index, src in enumerate(todo, 1):
        print(f"START {index}/{len(todo)} {src.name}", flush=True)
        record = convert_one(src)
        if record["status"] != "skipped":
            ledger.append(record)
        extra = ""
        if record["status"] == "ok":
            extra = f" pages={record.get('pages')} sec={record.get('seconds')} bytes={record.get('chars')}"
        else:
            extra = f" {record.get('error', '')[:180]}"
        print(f"{record['status'].upper()} {index}/{len(todo)} {src.name}{extra}", flush=True)
    print("CORPUS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
