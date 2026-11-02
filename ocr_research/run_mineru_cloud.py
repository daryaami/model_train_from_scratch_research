#!/usr/bin/env python3
"""Convert the chemistry corpus through the MinerU cloud API.

The API accepts at most 200 pages and 200 MB per request. Larger books
are cut into parts, then the Markdown is joined back into one file.
Finished books and finished parts are recorded in the shared manifest
and are not uploaded again. Run the same command to continue after a stop.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mineru_ledger as ledger
from mineru_api import MAX_BYTES, MAX_PAGES, MinerUError, document_to_markdown

ROOT = Path(__file__).resolve().parents[1]
SRC = ledger.SRC
DJVU_BIN = ROOT / ".tools" / "djvulibre" / "usr" / "bin"
DDJVU = DJVU_BIN / "ddjvu"
DJVUSED = DJVU_BIN / "djvused"
DJVUTXT = DJVU_BIN / "djvutxt"
EXTENSIONS = {".djvu", ".djv", ".pdf"}
CYRILLIC = re.compile(r"[А-Яа-яЁё]")
MIN_CYRILLIC_LETTERS = 20


class StopRun(RuntimeError):
    """The API asked us to stop, usually because today's quota is spent."""


class SplitNeeded(RuntimeError):
    """This slice is still too large or the request timed out."""


def contains_cyrillic(path: Path) -> bool:
    """True when the book name or a text sample contains Cyrillic.

    A book already cut into parts is kept so an in-progress Russian
    textbook is finished. Latin-only names with an empty text layer
    are left for later.
    """
    if CYRILLIC.search(path.name) or "(ru)" in path.name.lower():
        return True
    parts = ledger.canonical_output(path).with_suffix(".parts")
    if parts.is_dir() and any(parts.glob("*.md")):
        return True
    if path.suffix.lower() == ".pdf":
        command = ["pdftotext", "-f", "1", "-l", "8", "-q", str(path), "-"]
    else:
        command = [str(DJVUTXT), "--page=1-8", str(path)]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=90,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return _cyrillic_letters(result.stdout or b"") >= MIN_CYRILLIC_LETTERS


def _cyrillic_letters(raw: bytes) -> int:
    as_utf8 = raw.decode("utf-8", errors="replace")
    utf8_count = len(CYRILLIC.findall(as_utf8))
    if utf8_count >= MIN_CYRILLIC_LETTERS:
        return utf8_count
    as_cp1251 = raw.decode("cp1251", errors="replace")
    return max(utf8_count, len(CYRILLIC.findall(as_cp1251)))


def sources() -> list[Path]:
    files = [
        path
        for path in SRC.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    ]
    files.sort()
    return files


def page_count(path: Path) -> int:
    if path.suffix.lower() == ".pdf":
        result = subprocess.run(
            ["pdfinfo", str(path)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Pages:"):
                return int(line.split()[1])
        return 0
    result = subprocess.run(
        [str(DJVUSED), str(path), "-e", "n"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    text = result.stdout.strip()
    return int(text) if text.isdigit() else 0


def iter_ranges(pages: int, limit: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 1
    while start <= pages:
        end = min(pages, start + limit - 1)
        ranges.append((start, end))
        start = end + 1
    return ranges


def render_djvu(src: Path, start: int, end: int, dest: Path) -> None:
    timeout = max(180, (end - start + 1) * 8 + 120)
    result = subprocess.run(
        [
            str(DDJVU),
            f"-page={start}-{end}",
            "-format=pdf",
            "-size=1700x2400",
            str(src),
            str(dest),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    stderr = ((result.stderr or "") + "\n" + (result.stdout or "")).strip()
    produced = dest.is_file() and dest.stat().st_size > 1000
    failed = (not produced) or result.returncode != 0 or "Cannot decode page" in stderr
    if failed:
        lines = [
            line
            for line in stderr.splitlines()
            if "Cannot decode" in line or line.startswith("ddjvu:")
        ]
        detail = "\n".join(lines[-6:]) or stderr[-500:] or "ddjvu failed"
        raise RuntimeError(detail)


def write_pdf_slice(src: Path, start: int, end: int, dest: Path) -> None:
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(src), strict=False)
    if reader.is_encrypted:
        reader.decrypt("")
    writer = PdfWriter()
    for index in range(start - 1, end):
        writer.add_page(reader.pages[index])
    with dest.open("wb") as handle:
        writer.write(handle)


def materialize(src: Path, start: int, end: int, directory: Path) -> Path:
    dest = directory / "slice.pdf"
    if src.suffix.lower() == ".pdf":
        write_pdf_slice(src, start, end, dest)
    else:
        render_djvu(src, start, end, dest)
    if not dest.is_file() or dest.stat().st_size == 0:
        raise RuntimeError(f"empty slice {start}-{end}")
    return dest


def page_budget(src: Path, pages: int) -> int:
    """Largest part size that should fit under both API limits."""
    if src.suffix.lower() == ".pdf":
        per_page = max(1, src.stat().st_size / pages)
        return max(1, min(MAX_PAGES, int((MAX_BYTES * 0.85) / per_page)))
    probe_pages = min(3, pages)
    with tempfile.TemporaryDirectory(prefix="mineru_probe_") as tmp:
        dest = Path(tmp) / "probe.pdf"
        render_djvu(src, 1, probe_pages, dest)
        per_page = max(1, dest.stat().st_size / probe_pages)
    return max(1, min(MAX_PAGES, int((MAX_BYTES * 0.8) / per_page)))


def part_path(src: Path, start: int, end: int) -> Path:
    final = ledger.canonical_output(src)
    return final.with_suffix(".parts") / f"{start:04d}-{end:04d}.md"


def timeout_for(pages: int) -> float:
    return max(1200, pages * 20)


def status_line(name: str, label: str, message: str) -> None:
    print(f"  {name} {label}: {message}", flush=True)


def is_daily_limit(exc: BaseException) -> bool:
    text = str(exc)
    return "-60018" in text or "-60019" in text


def is_transient(exc: BaseException) -> bool:
    text = str(exc)
    markers = ("-60001", "-60007", "-60009", "-60010", "-10001", "-60022", "request failed")
    return any(marker in text for marker in markers)


def is_split_error(exc: BaseException) -> bool:
    text = str(exc)
    return "-60005" in text or "-60006" in text or "timed out" in text


def record_error(src: Path, message: str, page_ranges: str | None = None) -> None:
    record = {
        "source": ledger.source_key(src),
        "output": ledger.output_field(ledger.canonical_output(src)),
        "backend": "cloud",
        "status": "error",
        "error": message[:1000],
        "started": ledger.now(),
        "finished": ledger.now(),
    }
    if page_ranges:
        record["page_ranges"] = page_ranges
    ledger.append(record)


def upload_slice(
    src: Path,
    pdf: Path,
    destination: Path,
    *,
    label: str | None,
    pages: int,
) -> None:
    delay = 30
    last: MinerUError | None = None
    for attempt in range(4):
        try:
            document_to_markdown(
                pdf,
                output=destination,
                ledger_source=src,
                ledger_pages=label,
                timeout=timeout_for(pages),
                on_status=lambda message, name=src.name, shown=label or "full": status_line(name, shown, message),
            )
            return
        except MinerUError as exc:
            last = exc
            if is_daily_limit(exc):
                raise StopRun(str(exc)) from exc
            if is_split_error(exc) and pages > 1:
                raise SplitNeeded(str(exc)) from exc
            if is_transient(exc) and attempt < 3:
                print(f"  retry {attempt + 1} {src.name}: {exc}"[:240], flush=True)
                time.sleep(delay)
                delay *= 2
                continue
            raise
    if last is not None:
        raise last


def process_span(src: Path, start: int, end: int) -> list[Path]:
    label = f"{start}-{end}"
    existing = ledger.finished_output(src, label, exact=True)
    if existing is not None:
        print(f"  part {label} already done", flush=True)
        return [existing]
    destination = part_path(src, start, end)
    print(f"  part {label}", flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix="mineru_cloud_") as tmp:
            pdf = materialize(src, start, end, Path(tmp))
            size = pdf.stat().st_size
            if size > MAX_BYTES:
                raise SplitNeeded(f"slice is {size / 1024 ** 2:.0f} MB")
            upload_slice(src, pdf, destination, label=label, pages=end - start + 1)
    except SplitNeeded as exc:
        return _split_span(src, start, end, exc)
    except RuntimeError as exc:
        if _bad_page(exc):
            if start < end:
                return _split_span(src, start, end, exc)
            record_error(src, str(exc), label)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                f"\n\n<!-- page {start} was not decoded -->\n\n",
                encoding="utf-8",
            )
            print(f"  skip undecoded page {start}", flush=True)
            return [destination]
        raise
    return [destination]


def _bad_page(exc: BaseException) -> bool:
    text = str(exc)
    return "Cannot decode page" in text or text.startswith("ddjvu:")


def _split_span(src: Path, start: int, end: int, exc: BaseException) -> list[Path]:
    if start >= end:
        raise RuntimeError(str(exc)) from exc
    mid = (start + end) // 2
    print(f"  split {start}-{end}: {exc}"[:240], flush=True)
    return process_span(src, start, mid) + process_span(src, mid + 1, end)


def merge_parts(src: Path, parts: list[Path], pages: int, started_at: str, started: float) -> None:
    dest = ledger.canonical_output(src)
    chunks: list[str] = []
    image_dirs: list[Path] = []
    for part in parts:
        text = part.read_text(encoding="utf-8").strip()
        if text:
            chunks.append(text)
        images = part.parent / "images"
        if images.is_dir():
            image_dirs.append(images)
    body = "\n\n".join(chunks).strip()
    if len(body) < ledger.MIN_OUTPUT_BYTES:
        raise RuntimeError("merged markdown is empty")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(body + "\n", encoding="utf-8")
    copied: set[Path] = set()
    for images in image_dirs:
        resolved = images.resolve()
        if resolved in copied:
            continue
        copied.add(resolved)
        shutil.copytree(images, dest.parent / "images", dirs_exist_ok=True)
    ledger.append(
        {
            "source": ledger.source_key(src),
            "output": ledger.output_field(dest),
            "backend": "cloud",
            "status": "ok",
            "pages": pages,
            "parts": len(parts),
            "chars": dest.stat().st_size,
            "seconds": round(time.monotonic() - started, 1),
            "started": started_at,
            "finished": ledger.now(),
        }
    )


def upload_whole(src: Path, pdf: Path, pages: int) -> None:
    upload_slice(
        src,
        pdf,
        ledger.canonical_output(src),
        label=None,
        pages=pages,
    )


def convert_book(src: Path) -> str:
    if ledger.finished_output(src) is not None:
        return "skip"
    started_at = ledger.now()
    started = time.monotonic()
    try:
        pages = page_count(src)
    except (subprocess.TimeoutExpired, OSError) as exc:
        record_error(src, f"page count failed: {exc}")
        return "error"
    if pages <= 0:
        record_error(src, "could not read the page count")
        return "error"
    print(f"  pages={pages}", flush=True)
    try:
        budget = page_budget(src, pages)
    except (subprocess.TimeoutExpired, OSError, RuntimeError) as exc:
        budget = min(20, pages)
        print(f"  page budget failed, using {budget}-page parts: {exc}"[:240], flush=True)
    if budget < pages or src.suffix.lower() != ".pdf" or src.stat().st_size > MAX_BYTES:
        print(f"  parts of up to {budget} pages", flush=True)
    ranges = iter_ranges(pages, budget)
    if len(ranges) == 1 and src.suffix.lower() == ".pdf" and src.stat().st_size <= MAX_BYTES:
        try:
            upload_whole(src, src, pages)
            return "ok"
        except StopRun:
            raise
        except SplitNeeded as exc:
            print(f"  full file needs a smaller slice: {exc}"[:240], flush=True)
            ranges = [(1, pages // 2), (pages // 2 + 1, pages)] if pages > 1 else ranges
        except MinerUError as exc:
            print(f"ERROR {src.name}: {exc}", flush=True)
            return "error"
    if len(ranges) == 1 and src.suffix.lower() in {".djvu", ".djv"}:
        try:
            with tempfile.TemporaryDirectory(prefix="mineru_cloud_") as tmp:
                pdf = materialize(src, 1, pages, Path(tmp))
                if pdf.stat().st_size <= MAX_BYTES:
                    upload_whole(src, pdf, pages)
                    return "ok"
            print("  rendered pdf is over 200 MB, splitting", flush=True)
        except StopRun:
            raise
        except SplitNeeded as exc:
            print(f"  full file needs a smaller slice: {exc}"[:240], flush=True)
        except MinerUError as exc:
            print(f"ERROR {src.name}: {exc}", flush=True)
            return "error"
        except (subprocess.TimeoutExpired, OSError, RuntimeError) as exc:
            print(f"  full render failed, trying smaller parts: {exc}"[:240], flush=True)
            if pages <= 1:
                record_error(src, str(exc))
                return "error"
        if pages > 1 and len(ranges) == 1:
            ranges = [(1, pages // 2), (pages // 2 + 1, pages)]
    parts: list[Path] = []
    failed = False
    for start, end in ranges:
        try:
            parts.extend(process_span(src, start, end))
        except StopRun:
            raise
        except Exception as exc:
            failed = True
            print(f"ERROR {src.name} {start}-{end}: {exc}", flush=True)
            if not isinstance(exc, MinerUError):
                record_error(src, str(exc), f"{start}-{end}")
    if failed or not parts:
        return "error"
    try:
        merge_parts(src, parts, pages, started_at, started)
    except Exception as exc:  # noqa: BLE001
        record_error(src, f"merge failed: {exc}")
        print(f"ERROR {src.name} merge: {exc}", flush=True)
        return "error"
    return "ok"


def main() -> int:
    if not SRC.is_dir():
        print(f"missing source dir: {SRC}", file=sys.stderr)
        return 1
    if not DDJVU.is_file() or not DJVUSED.is_file():
        print(f"missing ddjvu tools in {DJVU_BIN}", file=sys.stderr)
        return 1
    files = sources()
    done = ledger.completed_sources()
    pending = [path for path in files if ledger.source_key(path) not in done]
    todo = []
    skipped = 0
    for path in pending:
        if contains_cyrillic(path):
            todo.append(path)
        else:
            skipped += 1
            print(f"SKIP latin {path.name}", flush=True)
    print(
        f"corpus files={len(files)} already_ok={len(done & {ledger.source_key(path) for path in files})} "
        f"cyrillic={len(todo)} skip_latin={skipped}",
        flush=True,
    )
    for index, src in enumerate(todo, 1):
        print(f"START {index}/{len(todo)} {src.name}", flush=True)
        started = time.monotonic()
        try:
            status = convert_book(src)
        except StopRun as exc:
            print(f"STOP {exc}", flush=True)
            print("rerun the same command to continue", flush=True)
            return 0
        except KeyboardInterrupt:
            print("STOP interrupted, rerun to continue", flush=True)
            return 0
        except Exception as exc:  # noqa: BLE001
            record_error(src, str(exc))
            print(f"ERROR {src.name}: {exc}", flush=True)
            continue
        elapsed = round(time.monotonic() - started, 1)
        print(f"{status.upper()} {index}/{len(todo)} {src.name} sec={elapsed}", flush=True)
    print("CLOUD_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
