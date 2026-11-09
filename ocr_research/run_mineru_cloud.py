#!/usr/bin/env python3
"""Convert the chemistry corpus through the MinerU cloud API.

The API accepts at most 200 pages and 200 MB per request. Larger books
are cut into parts, then the Markdown is joined back into one file.
Finished books and finished parts are recorded in the shared manifest
and are not uploaded again. Several books are sent at once with one API key.
A failed book is retried once, then skipped. Daily quota pauses that key
until the next Shanghai midnight; the other key keeps working.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from queue import Queue
from pathlib import Path
from zoneinfo import ZoneInfo

import mineru_ledger as ledger
from mineru_api import MAX_BYTES, MAX_PAGES, MinerUError, document_to_markdown, load_api_keys

ROOT = Path(__file__).resolve().parents[1]
SRC = ledger.SRC
CHEM_TEXT = ROOT / "data" / "chem_dataset" / "chem_text"
DJVU_BIN = ROOT / ".tools" / "djvulibre" / "usr" / "bin"
DDJVU = DJVU_BIN / "ddjvu"
DJVUSED = DJVU_BIN / "djvused"
DJVUTXT = DJVU_BIN / "djvutxt"
EXTENSIONS = {".djvu", ".djv", ".pdf"}
CYRILLIC = re.compile(r"[А-Яа-яЁё]")
LATIN = re.compile(r"[A-Za-z]")
MIN_CYRILLIC_LETTERS = 20
PRINT_LOCK = threading.Lock()
WORKERS = 6
API_SLOTS = threading.BoundedSemaphore(WORKERS)
_TLS = threading.local()
STOP_ON_ERROR = False
_KEY_LOCK = threading.Lock()
_KEY_PAUSE_UNTIL: dict[str, float] = {}
_DEAD_KEYS: set[str] = set()
_ALL_KEY_NAMES: list[str] = []
_RETRIED: set[str] = set()
_RETRIED_LOCK = threading.Lock()


def set_workers(count: int) -> None:
    global WORKERS, API_SLOTS
    WORKERS = max(1, count)
    API_SLOTS = threading.BoundedSemaphore(WORKERS)


class StopRun(RuntimeError):
    """This API key cannot continue (auth or missing credentials)."""


class QuotaPause(RuntimeError):
    """This key hit today's page/task cap; wait and retry."""


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
    if _chem_text_is_russian(path):
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


def _chem_text_is_russian(path: Path) -> bool:
    """True when chem_text already has a Russian Tesseract/text-layer dump."""
    try:
        relative = path.resolve().relative_to(SRC.resolve())
    except ValueError:
        return False
    txt = CHEM_TEXT / relative.with_suffix(".txt")
    if not txt.is_file():
        return False
    sample = txt.read_text(encoding="utf-8", errors="replace")[:200_000]
    cyr = len(CYRILLIC.findall(sample))
    lat = len(LATIN.findall(sample))
    total = cyr + lat
    return total > 0 and cyr / total >= 0.5


def tesseract_pending() -> set[str]:
    """Russian books that Tesseract already dumped and MinerU has not finished."""
    manifest = CHEM_TEXT / "_manifest.jsonl"
    if not manifest.is_file():
        return set()
    latest: dict[str, dict] = {}
    for record in ledger.iter_records(manifest):
        source = record.get("source")
        if source:
            latest[str(source)] = record
    done = ledger.completed_sources()
    pending: set[str] = set()
    for source, record in latest.items():
        if record.get("status") != "ok" or record.get("method") != "tesseract":
            continue
        if source in done:
            continue
        output = record.get("output")
        txt = CHEM_TEXT / output if output else CHEM_TEXT / Path(source).with_suffix(".txt")
        if not txt.is_file():
            continue
        pending.add(source)
    return pending


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


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def status_line(name: str, label: str, message: str) -> None:
    log(f"  {name} {label}: {message}")


def is_daily_limit(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "-60018" in text or "-60019" in text or (
        "daily" in text and ("limit" in text or "quota" in text or "exceed" in text)
    )


def is_auth_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("unauthorized", "invalid token", "invalid api", "401", "403 forbidden")
    )


def is_transient(exc: BaseException) -> bool:
    text = str(exc)
    markers = (
        "-60001",
        "-60007",
        "-60009",
        "-60010",
        "-10001",
        "-60022",
        "request failed",
        "parsing failed",
        "name resolution",
        "EOF occurred",
        "violation of protocol",
        "SSL",
        "ssl",
        "Connection reset",
        "Connection aborted",
        "RemoteDisconnected",
        "Broken pipe",
        "timed out",
        "timeout",
        "429",
        "502",
        "503",
        "504",
    )
    return any(marker in text for marker in markers)


def seconds_until_quota_reset() -> float:
    try:
        tz = ZoneInfo("Asia/Shanghai")
    except Exception:
        tz = timezone.utc
    now = datetime.now(tz)
    nxt = (now + timedelta(days=1)).replace(hour=0, minute=20, second=0, microsecond=0)
    return max(600.0, (nxt - now).total_seconds())


def pause_key(key_name: str, seconds: float | None = None) -> None:
    wait = seconds if seconds is not None else seconds_until_quota_reset()
    until = time.monotonic() + wait
    with _KEY_LOCK:
        _KEY_PAUSE_UNTIL[key_name] = max(_KEY_PAUSE_UNTIL.get(key_name, 0), until)
    log(f"PAUSE [{key_name}] {wait / 3600:.1f}h until quota retry")


def wait_for_key(key_name: str) -> None:
    while True:
        with _KEY_LOCK:
            if key_name in _DEAD_KEYS:
                return
            remaining = _KEY_PAUSE_UNTIL.get(key_name, 0) - time.monotonic()
        if remaining <= 0:
            return
        log(f"WAIT [{key_name}] quota {remaining / 60:.0f} min left")
        time.sleep(min(remaining, 300))


def kill_key(key_name: str) -> None:
    with _KEY_LOCK:
        _DEAD_KEYS.add(key_name)
    log(f"DISABLE [{key_name}] will not send more requests")


def is_key_dead(key_name: str) -> bool:
    with _KEY_LOCK:
        return key_name in _DEAD_KEYS


def all_keys_dead() -> bool:
    with _KEY_LOCK:
        return bool(_ALL_KEY_NAMES) and all(name in _DEAD_KEYS for name in _ALL_KEY_NAMES)


def should_requeue(src: Path) -> bool:
    key = ledger.source_key(src)
    with _RETRIED_LOCK:
        if key in _RETRIED:
            return False
        _RETRIED.add(key)
        return True


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
    last: BaseException | None = None
    attempts = 6
    for attempt in range(attempts):
        try:
            with API_SLOTS:
                document_to_markdown(
                    pdf,
                    output=destination,
                    ledger_source=src,
                    ledger_pages=label,
                    timeout=timeout_for(pages),
                    api_key=getattr(_TLS, "api_key", None),
                    on_status=lambda message, name=src.name, shown=label or "full": status_line(
                        name, shown, message
                    ),
                )
            return
        except Exception as exc:
            last = exc
            if is_daily_limit(exc):
                raise QuotaPause(str(exc)) from exc
            if is_auth_error(exc):
                raise StopRun(str(exc)) from exc
            if is_split_error(exc) and pages > 1:
                raise SplitNeeded(str(exc)) from exc
            if is_transient(exc) and attempt < attempts - 1:
                log(f"  retry {attempt + 1} {src.name}: {exc}"[:240])
                time.sleep(delay)
                delay = min(delay * 2, 600)
                continue
            raise
    if last is not None:
        raise last


def process_span(src: Path, start: int, end: int) -> list[Path]:
    label = f"{start}-{end}"
    existing = ledger.finished_output(src, label, exact=True)
    if existing is not None:
        log(f"  part {label} already done")
        return [existing]
    destination = part_path(src, start, end)
    log(f"  part {label}")
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
            log(f"  skip undecoded page {start}")
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
    log(f"  split {start}-{end}: {exc}"[:240])
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
    log(f"  pages={pages}")
    try:
        budget = page_budget(src, pages)
    except (subprocess.TimeoutExpired, OSError, RuntimeError) as exc:
        budget = min(20, pages)
        log(f"  page budget failed, using {budget}-page parts: {exc}"[:240])
    if budget < pages or src.suffix.lower() != ".pdf" or src.stat().st_size > MAX_BYTES:
        log(f"  parts of up to {budget} pages")
    ranges = iter_ranges(pages, budget)
    if len(ranges) == 1 and src.suffix.lower() == ".pdf" and src.stat().st_size <= MAX_BYTES:
        try:
            upload_whole(src, src, pages)
            return "ok"
        except (StopRun, QuotaPause):
            raise
        except SplitNeeded as exc:
            log(f"  full file needs a smaller slice: {exc}"[:240])
            ranges = [(1, pages // 2), (pages // 2 + 1, pages)] if pages > 1 else ranges
        except MinerUError as exc:
            log(f"ERROR {src.name}: {exc}")
            return "error"
    if len(ranges) == 1 and src.suffix.lower() in {".djvu", ".djv"}:
        try:
            with tempfile.TemporaryDirectory(prefix="mineru_cloud_") as tmp:
                pdf = materialize(src, 1, pages, Path(tmp))
                if pdf.stat().st_size <= MAX_BYTES:
                    upload_whole(src, pdf, pages)
                    return "ok"
            log("  rendered pdf is over 200 MB, splitting")
        except (StopRun, QuotaPause):
            raise
        except SplitNeeded as exc:
            log(f"  full file needs a smaller slice: {exc}"[:240])
        except MinerUError as exc:
            log(f"ERROR {src.name}: {exc}")
            return "error"
        except (subprocess.TimeoutExpired, OSError, RuntimeError) as exc:
            log(f"  full render failed, trying smaller parts: {exc}"[:240])
            if pages <= 1:
                record_error(src, str(exc))
                return "error"
        if pages > 1 and len(ranges) == 1:
            ranges = [(1, pages // 2), (pages // 2 + 1, pages)]
    parts: list[Path] = []
    for start, end in ranges:
        try:
            parts.extend(process_span(src, start, end))
        except (StopRun, QuotaPause):
            raise
        except Exception as exc:
            log(f"ERROR {src.name} {start}-{end}: {exc}")
            if not isinstance(exc, MinerUError):
                record_error(src, str(exc), f"{start}-{end}")
            return "error"
    if not parts:
        return "error"
    try:
        merge_parts(src, parts, pages, started_at, started)
    except Exception as exc:  # noqa: BLE001
        record_error(src, f"merge failed: {exc}")
        log(f"ERROR {src.name} merge: {exc}")
        return "error"
    return "ok"


def _run_one(src: Path, index: int, total: int, key_name: str) -> tuple[str, float]:
    log(f"START {index}/{total} [{key_name}] {src.name}")
    started = time.monotonic()
    status = convert_book(src)
    return status, round(time.monotonic() - started, 1)


def _requeue(jobs: Queue, item: tuple[int, Path], src: Path, why: str) -> None:
    if should_requeue(src):
        log(f"REQUEUE {src.name} ({why})")
        jobs.put(item)
    else:
        log(f"DROP {src.name} after retry ({why})")


def _key_worker(
    key_name: str,
    key: str,
    jobs: Queue,
    total: int,
    stop: threading.Event,
    reason: list[str],
) -> None:
    _TLS.api_key = key
    _TLS.key_name = key_name
    while True:
        item = jobs.get()
        try:
            if item is None:
                return
            if stop.is_set():
                continue
            if is_key_dead(key_name):
                jobs.put(item)
                while is_key_dead(key_name) and not stop.is_set():
                    time.sleep(5)
                continue
            wait_for_key(key_name)
            if stop.is_set() or is_key_dead(key_name):
                jobs.put(item)
                continue
            index, src = item
            try:
                status, elapsed = _run_one(src, index, total, key_name)
            except QuotaPause as exc:
                log(f"QUOTA [{key_name}] {src.name}: {exc}"[:240])
                pause_key(key_name)
                jobs.put(item)
                continue
            except StopRun as exc:
                log(f"AUTH [{key_name}] {src.name}: {exc}"[:240])
                kill_key(key_name)
                jobs.put(item)
                if all_keys_dead():
                    reason.append("all keys disabled")
                    stop.set()
                continue
            except Exception as exc:  # noqa: BLE001
                record_error(src, str(exc))
                log(f"ERROR [{key_name}] {src.name}: {exc}")
                if STOP_ON_ERROR:
                    reason.append("cloud error")
                    stop.set()
                else:
                    _requeue(jobs, item, src, "exception")
                continue
            log(f"{status.upper()} {index}/{total} [{key_name}] {src.name} sec={elapsed}")
            if status == "error":
                if STOP_ON_ERROR:
                    reason.append("cloud error")
                    stop.set()
                else:
                    _requeue(jobs, item, src, "error")
        finally:
            jobs.task_done()


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert the chemistry corpus with the MinerU cloud API.")
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="How many API tasks to keep in flight per key",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Exit the whole run on the first failed book (old behaviour)",
    )
    args = parser.parse_args()
    global STOP_ON_ERROR, _ALL_KEY_NAMES
    STOP_ON_ERROR = args.stop_on_error
    keys = load_api_keys()
    _ALL_KEY_NAMES = [name for name, _ in keys]
    set_workers(args.workers * len(keys))
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
            log(f"SKIP latin {path.name}")
    tess = tesseract_pending()
    todo.sort(key=lambda path: (0 if ledger.source_key(path) in tess else 1, path.as_posix()))
    log(
        f"corpus files={len(files)} already_ok={len(done & {ledger.source_key(path) for path in files})} "
        f"cyrillic={len(todo)} skip_latin={skipped} tesseract_first={sum(1 for path in todo if ledger.source_key(path) in tess)} "
        f"keys={len(keys)} workers_per_key={args.workers}"
    )
    jobs: Queue = Queue()
    for index, src in enumerate(todo, 1):
        jobs.put((index, src))
    stop = threading.Event()
    reason: list[str] = []
    threads: list[threading.Thread] = []
    for key_name, key in keys:
        for _ in range(args.workers):
            thread = threading.Thread(
                target=_key_worker,
                args=(key_name, key, jobs, len(todo), stop, reason),
                daemon=True,
            )
            threads.append(thread)
            thread.start()
    try:
        jobs.join()
    except KeyboardInterrupt:
        reason.append("interrupted")
        stop.set()
    stop.set()
    for _ in threads:
        jobs.put(None)
    for thread in threads:
        thread.join(timeout=15)
    stop_reason = reason[0] if reason else None
    if stop_reason == "cloud error":
        log("STOP on cloud error, rerun the same command to continue")
        return 1
    if stop_reason == "interrupted":
        log("STOP interrupted, rerun to continue")
        return 0
    if stop_reason:
        log(f"STOP {stop_reason}")
        log("rerun the same command to continue")
        return 0
    log("CLOUD_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
