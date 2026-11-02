"""Shared record of documents MinerU has already converted.

Local mineru-kit runs and the cloud API append to the same jsonl file.
A later run skips a document when an earlier successful record points at
a Markdown file that is still on disk. A partial page range does not
count as the whole document.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "chem_dataset" / "chem_extracted"
OUT = ROOT / "data" / "chem_dataset" / "chem_text_mineru"
MANIFEST = OUT / "_manifest.jsonl"
MIN_OUTPUT_BYTES = 20


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_key(source: str | Path) -> str:
    if isinstance(source, str) and source.startswith(("http://", "https://")):
        return source
    path = Path(source)
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        return path.resolve().relative_to(SRC.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def canonical_output(source: str | Path, output: Path | None = None) -> Path:
    if output is not None:
        return output
    key = source_key(source)
    if key.startswith(("http://", "https://")):
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return OUT / "_urls" / f"{digest}.md"
    path = Path(key)
    if path.is_absolute():
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return OUT / "_external" / f"{path.stem}-{digest}.md"
    return OUT / path.with_suffix(".md")


def output_field(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(OUT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def resolve_output(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return OUT / path


def append(record: dict, manifest: Path | None = None) -> None:
    path = MANIFEST if manifest is None else manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def iter_records(manifest: Path | None = None):
    path = MANIFEST if manifest is None else manifest
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def _latest_ok(manifest: Path | None = None) -> dict[tuple[str, str], str]:
    found: dict[tuple[str, str], str] = {}
    for record in iter_records(manifest):
        if record.get("status") != "ok":
            continue
        key = record.get("source")
        output = record.get("output")
        if not key or not output:
            continue
        pages = str(record.get("page_ranges") or "")
        found[(str(key), pages)] = str(output)
    return found


def _usable(path: Path | None) -> bool:
    return path is not None and path.is_file() and path.stat().st_size >= MIN_OUTPUT_BYTES


def finished_output(
    source: str | Path,
    page_ranges: str | None = None,
    manifest: Path | None = None,
    *,
    exact: bool = False,
) -> Path | None:
    """Markdown already produced for this document, if it should be reused.

    A finished full document satisfies a later page-range request.
    A finished page range satisfies only that same range.
    ``exact`` does not let a full document stand in for one page range.
    """
    key = source_key(source)
    want = page_ranges or ""
    latest = _latest_ok(manifest)
    if exact:
        candidates = (want,)
    else:
        candidates = ("", want) if want else ("",)
    for pages in candidates:
        path = resolve_output(latest.get((key, pages)))
        if _usable(path):
            return path
    return None


def completed_sources(manifest: Path | None = None) -> set[str]:
    """Sources whose full document was converted and whose Markdown remains."""
    done: set[str] = set()
    for (key, pages), output in _latest_ok(manifest).items():
        if pages:
            continue
        if _usable(resolve_output(output)):
            done.add(key)
    return done
