#!/usr/bin/env python3
"""Convert a document to Markdown through the MinerU precise API.

The token is read from MINERU_API_KEY in the environment or in the
repository .env file. Local files are uploaded, remote http(s) URLs are
submitted as-is. The call waits until parsing finishes and returns the
Markdown from full.md inside the result archive.

Successful conversions are appended to the same manifest the local
mineru-kit corpus run uses. A document already converted by either path
is returned from that Markdown and is not sent to the API again.

Limits of this API: 200 MB and 200 pages per file. DjVu is not accepted;
render it to PDF first. HTML is sent with model_version MinerU-HTML.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import mineru_ledger as ledger

ROOT = Path(__file__).resolve().parents[1]
API = "https://mineru.net"
MAX_BYTES = 200 * 1024 * 1024
MAX_PAGES = 200
UPLOAD_SUFFIXES = {
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".png",
    ".jpg",
    ".jpeg",
    ".jp2",
    ".webp",
    ".gif",
    ".bmp",
    ".html",
    ".htm",
}


class MinerUError(RuntimeError):
    """MinerU rejected the request or the task failed."""


def document_to_markdown(
    source: str | Path,
    *,
    output: str | Path | None = None,
    model_version: str = "vlm",
    language: str = "east_slavic",
    is_ocr: bool = True,
    enable_formula: bool = True,
    enable_table: bool = True,
    page_ranges: str | None = None,
    timeout: float = 900,
    poll_interval: float = 5,
    on_status: Callable[[str], None] | None = None,
    force: bool = False,
    ledger_source: str | Path | None = None,
    ledger_pages: str | None = None,
) -> str:
    """Convert a local document or an http(s) URL to Markdown.

    ``source`` is a filesystem path or a URL MinerU can download.
    ``language`` defaults to ``east_slavic`` (Russian, Belarusian,
    Ukrainian). ``model_version`` ``vlm`` is the accurate model; use
    ``pipeline`` for the lighter one. The Markdown is written to
    ``output`` or, when that is omitted, next to the other MinerU
    results in ``data/chem_dataset/chem_text_mineru``. Images from the
    result archive are copied beside it.

    A document that already has a successful manifest record and a
    Markdown file on disk is not uploaded again. ``force`` processes
    it once more and appends a new record.

    ``ledger_source`` and ``ledger_pages`` record a slice under the
    original document. ``page_ranges`` is what the API receives for the
    uploaded file.

    Returns the Markdown text.
    """
    identity = ledger_source if ledger_source is not None else source
    recorded_pages = ledger_pages if ledger_pages is not None else page_ranges
    destination = ledger.canonical_output(identity, Path(output) if output is not None else None)
    if not force:
        existing = ledger.finished_output(
            identity,
            recorded_pages,
            exact=ledger_pages is not None,
        )
        if existing is not None:
            _status(on_status, f"already processed, skip {ledger.source_key(identity)}")
            target = destination if output is not None else existing
            return _reuse(existing, target)
    started_at = ledger.now()
    started = time.monotonic()
    directory: Path | None = None
    try:
        client = _Client(_api_key(), on_status)
        text, md_path, directory = client.convert(
            source,
            model_version=model_version,
            language=language,
            is_ocr=is_ocr,
            enable_formula=enable_formula,
            enable_table=enable_table,
            page_ranges=page_ranges,
            timeout=timeout,
            poll_interval=poll_interval,
        )
        _write_output(text, md_path, destination)
        record = {
            "source": ledger.source_key(identity),
            "output": ledger.output_field(destination),
            "backend": "cloud",
            "model_version": model_version,
            "status": "ok",
            "chars": destination.stat().st_size,
            "seconds": round(time.monotonic() - started, 1),
            "started": started_at,
            "finished": ledger.now(),
        }
        if recorded_pages:
            record["page_ranges"] = recorded_pages
        pages = _page_count(source)
        if pages is not None:
            record["pages"] = pages
        ledger.append(record)
        return text
    except Exception as exc:
        error = {
            "source": ledger.source_key(identity),
            "output": ledger.output_field(destination),
            "backend": "cloud",
            "model_version": model_version,
            "status": "error",
            "error": str(exc)[:1000],
            "seconds": round(time.monotonic() - started, 1),
            "started": started_at,
            "finished": ledger.now(),
        }
        if recorded_pages:
            error["page_ranges"] = recorded_pages
        ledger.append(error)
        raise
    finally:
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)


def _api_key() -> str:
    key = os.environ.get("MINERU_API_KEY", "").strip().strip('"').strip("'")
    if key:
        return key
    env_path = ROOT / ".env"
    if env_path.is_file():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "MINERU_API_KEY":
                key = value.strip().strip('"').strip("'")
                break
    if not key:
        raise MinerUError("MINERU_API_KEY is missing from the environment and from .env")
    return key


def _status(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


class _Client:
    def __init__(self, token: str, on_status: Callable[[str], None] | None) -> None:
        self._headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        self._on_status = on_status

    def convert(
        self,
        source: str | Path,
        *,
        model_version: str,
        language: str,
        is_ocr: bool,
        enable_formula: bool,
        enable_table: bool,
        page_ranges: str | None,
        timeout: float,
        poll_interval: float,
    ) -> tuple[str, Path, Path]:
        remote = _is_url(source)
        version = model_version
        if remote and Path(urlparse(str(source)).path).suffix.lower() in {".html", ".htm"}:
            version = "MinerU-HTML"
        if remote:
            task_id = self._submit_url(
                str(source),
                model_version=version,
                language=language,
                is_ocr=is_ocr,
                enable_formula=enable_formula,
                enable_table=enable_table,
                page_ranges=page_ranges,
            )
            zip_url = self._poll_task(task_id, timeout, poll_interval)
        else:
            path = Path(source)
            _check_local(path, page_ranges)
            version = "MinerU-HTML" if path.suffix.lower() in {".html", ".htm"} else model_version
            batch_id, file_name = self._upload(
                path,
                model_version=version,
                language=language,
                is_ocr=is_ocr,
                enable_formula=enable_formula,
                enable_table=enable_table,
                page_ranges=page_ranges,
            )
            zip_url = self._poll_batch(batch_id, file_name, timeout, poll_interval)
        directory = self._download_zip(zip_url)
        try:
            text, md_path = _markdown_from_zip(directory)
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        return text, md_path, directory

    def _submit_url(
        self,
        url: str,
        *,
        model_version: str,
        language: str,
        is_ocr: bool,
        enable_formula: bool,
        enable_table: bool,
        page_ranges: str | None,
    ) -> str:
        _status(self._on_status, "submitting url")
        body: dict = {
            "url": url,
            "model_version": model_version,
            "language": language,
            "is_ocr": is_ocr,
            "enable_formula": enable_formula,
            "enable_table": enable_table,
        }
        if page_ranges:
            body["page_ranges"] = page_ranges
        payload = self._json("POST", f"{API}/api/v4/extract/task", body)
        task_id = (payload.get("data") or {}).get("task_id")
        if not task_id:
            raise MinerUError("MinerU did not return task_id")
        return str(task_id)

    def _upload(
        self,
        path: Path,
        *,
        model_version: str,
        language: str,
        is_ocr: bool,
        enable_formula: bool,
        enable_table: bool,
        page_ranges: str | None,
    ) -> tuple[str, str]:
        file_name = path.name
        file_item: dict = {"name": file_name, "is_ocr": is_ocr}
        if page_ranges:
            file_item["page_ranges"] = page_ranges
        body = {
            "files": [file_item],
            "model_version": model_version,
            "language": language,
            "enable_formula": enable_formula,
            "enable_table": enable_table,
        }
        _status(self._on_status, f"requesting upload url for {file_name}")
        payload = self._json("POST", f"{API}/api/v4/file-urls/batch", body)
        data = payload.get("data") or {}
        batch_id = data.get("batch_id")
        urls = data.get("file_urls") or []
        if not batch_id or not urls:
            raise MinerUError("MinerU did not return an upload url")
        _status(self._on_status, f"uploading {file_name}")
        _put_file(str(urls[0]), path)
        return str(batch_id), file_name

    def _poll_task(self, task_id: str, timeout: float, poll_interval: float) -> str:
        deadline = time.monotonic() + timeout
        url = f"{API}/api/v4/extract/task/{task_id}"
        while True:
            payload = self._json("GET", url, None)
            data = payload.get("data") or {}
            state = str(data.get("state") or "")
            if state == "done":
                zip_url = data.get("full_zip_url")
                if not zip_url:
                    raise MinerUError("task is done but full_zip_url is missing")
                return str(zip_url)
            if state == "failed":
                raise MinerUError(data.get("err_msg") or "parsing failed")
            if time.monotonic() >= deadline:
                raise MinerUError(f"timed out after {timeout:.0f}s, last state: {state or 'unknown'}")
            progress = data.get("extract_progress") or {}
            _status(self._on_status, _progress_line(state, progress))
            time.sleep(poll_interval)

    def _poll_batch(
        self,
        batch_id: str,
        file_name: str,
        timeout: float,
        poll_interval: float,
    ) -> str:
        deadline = time.monotonic() + timeout
        url = f"{API}/api/v4/extract-results/batch/{batch_id}"
        while True:
            payload = self._json("GET", url, None)
            results = (payload.get("data") or {}).get("extract_result") or []
            item = _match_result(results, file_name)
            state = str(item.get("state") or "")
            if state == "done":
                zip_url = item.get("full_zip_url")
                if not zip_url:
                    raise MinerUError("task is done but full_zip_url is missing")
                return str(zip_url)
            if state == "failed":
                raise MinerUError(item.get("err_msg") or "parsing failed")
            if time.monotonic() >= deadline:
                raise MinerUError(f"timed out after {timeout:.0f}s, last state: {state or 'unknown'}")
            progress = item.get("extract_progress") or {}
            _status(self._on_status, _progress_line(state, progress))
            time.sleep(poll_interval)

    def _download_zip(self, zip_url: str) -> Path:
        _status(self._on_status, "downloading result")
        directory = Path(tempfile.mkdtemp(prefix="mineru_api_"))
        destination = directory / "result.zip"
        request = urllib.request.Request(zip_url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=300) as response, destination.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        except urllib.error.URLError as exc:
            shutil.rmtree(directory, ignore_errors=True)
            raise MinerUError(f"failed to download the result archive: {exc.reason}") from exc
        return directory

    def _json(self, method: str, url: str, body: dict | None) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=self._headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = response.read()
                status = response.status
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            status = exc.code
        except urllib.error.URLError as exc:
            raise MinerUError(f"MinerU request failed: {exc.reason}") from exc
        return _parse_payload(status, raw)


def _is_url(source: str | Path) -> bool:
    return isinstance(source, str) and source.startswith(("http://", "https://"))


def _check_local(path: Path, page_ranges: str | None) -> None:
    if not path.is_file():
        raise MinerUError(f"file not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in UPLOAD_SUFFIXES:
        raise MinerUError(
            f"unsupported type {suffix or '(none)'}. "
            "MinerU accepts pdf, office, html, and common images. "
            "DjVu has to be rendered to PDF first."
        )
    size = path.stat().st_size
    if size == 0:
        raise MinerUError(f"empty file: {path}")
    if size > MAX_BYTES:
        raise MinerUError(f"file is larger than 200 MB: {path}")
    if suffix == ".pdf" and not page_ranges:
        pages = _pdf_page_count(path)
        if pages is not None and pages > MAX_PAGES:
            raise MinerUError(
                f"PDF has {pages} pages, the API limit is {MAX_PAGES}. "
                "Pass page_ranges, for example '1-200'."
            )


def _pdf_page_count(path: Path) -> int | None:
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    try:
        return len(PdfReader(str(path)).pages)
    except Exception:
        return None


def _put_file(url: str, path: Path) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise MinerUError("MinerU returned an invalid upload url")
    target = parsed.path or "/"
    if parsed.query:
        target = f"{target}?{parsed.query}"
    connection_cls = HTTPSConnection if parsed.scheme == "https" else HTTPConnection
    timeout = max(180, int(path.stat().st_size / (512 * 1024)) + 60)
    connection = connection_cls(parsed.netloc, timeout=timeout)
    try:
        with path.open("rb") as handle:
            connection.request(
                "PUT",
                target,
                body=handle,
                headers={"Content-Length": str(path.stat().st_size)},
            )
            response = connection.getresponse()
            detail = response.read(500)
            status = response.status
    except OSError as exc:
        raise MinerUError(f"upload failed: {exc}") from exc
    finally:
        connection.close()
    if status < 200 or status >= 300:
        text = detail.decode("utf-8", errors="replace").strip()
        extra = f": {text}" if text else ""
        raise MinerUError(f"upload failed with HTTP {status}{extra}")


def _parse_payload(status: int, raw: bytes) -> dict:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MinerUError(f"MinerU returned a non-JSON response (HTTP {status})") from exc
    if not isinstance(payload, dict):
        raise MinerUError(f"MinerU returned an unexpected response (HTTP {status})")
    code = payload.get("code")
    if status >= 400 or code not in (0, "0", None):
        message = payload.get("msg") or payload.get("message") or "request failed"
        raise MinerUError(f"MinerU API error {code}: {message}")
    return payload


def _match_result(results: list, file_name: str) -> dict:
    if not isinstance(results, list) or not results:
        return {}
    for item in results:
        if isinstance(item, dict) and item.get("file_name") == file_name:
            return item
    first = results[0]
    return first if isinstance(first, dict) else {}


def _progress_line(state: str, progress: dict) -> str:
    done = progress.get("extracted_pages")
    total = progress.get("total_pages")
    if done is not None and total is not None:
        return f"{state}: {done}/{total} pages"
    return state or "waiting"


def _safe_extract(handle: zipfile.ZipFile, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    base = root.resolve()
    for info in handle.infolist():
        target = (root / info.filename).resolve()
        if target != base and base not in target.parents:
            raise MinerUError("result archive contains an unsafe path")
    handle.extractall(root)


def _markdown_from_zip(directory: Path) -> tuple[str, Path]:
    archive = directory / "result.zip"
    if not zipfile.is_zipfile(archive):
        raise MinerUError("MinerU result is not a zip archive")
    root = directory / "unzipped"
    with zipfile.ZipFile(archive) as handle:
        _safe_extract(handle, root)
    matches = [path for path in root.rglob("full.md") if path.is_file()]
    if not matches:
        matches = [path for path in root.rglob("*.md") if path.is_file()]
    if not matches:
        raise MinerUError("result archive does not contain a markdown file")
    chosen = min(matches, key=lambda path: (len(path.relative_to(root).parts), str(path)))
    return chosen.read_text(encoding="utf-8"), chosen


def _page_count(source: str | Path) -> int | None:
    if _is_url(source):
        return None
    path = Path(source)
    if path.suffix.lower() != ".pdf" or not path.is_file():
        return None
    return _pdf_page_count(path)


def _reuse(existing: Path, output: Path) -> str:
    text = existing.read_text(encoding="utf-8")
    if output.resolve() == existing.resolve():
        return text
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    _copy_images(existing.parent, output.parent, skip=existing)
    return text


def _write_output(text: str, md_path: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    _copy_images(md_path.parent, output.parent, skip=md_path)


def _copy_images(source_dir: Path, dest_dir: Path, skip: Path) -> None:
    image_suffixes = {".png", ".jpg", ".jpeg", ".jp2", ".webp", ".gif", ".bmp"}
    for item in source_dir.iterdir():
        if item.resolve() == skip.resolve():
            continue
        if not item.is_dir() and item.suffix.lower() not in image_suffixes:
            continue
        destination = dest_dir / item.name
        if destination.resolve() == item.resolve():
            continue
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Convert a document to Markdown with the MinerU API.")
    parser.add_argument("source", help="Local file path or http(s) URL")
    parser.add_argument("-o", "--output", help="Where to write the Markdown")
    parser.add_argument("--model", default="vlm", choices=["vlm", "pipeline", "MinerU-HTML"])
    parser.add_argument("--language", default="east_slavic")
    parser.add_argument("--pages", default=None, help="Page range, for example 1-20 or 2,4-6")
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--no-formula", action="store_true")
    parser.add_argument("--no-table", action="store_true")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--force", action="store_true", help="Process again even if this document is already done")
    args = parser.parse_args()
    text = document_to_markdown(
        args.source,
        output=args.output,
        model_version=args.model,
        language=args.language,
        is_ocr=not args.no_ocr,
        enable_formula=not args.no_formula,
        enable_table=not args.no_table,
        page_ranges=args.pages,
        timeout=args.timeout,
        on_status=print,
        force=args.force,
    )
    if not args.output:
        print(text)


if __name__ == "__main__":
    _main()
