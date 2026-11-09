#!/usr/bin/env python3
"""Preprocess a frozen MinerU pilot without touching the source Markdown.

Reads configs/preprocess_v001.yaml. Copies the chosen books once into
chem_preprocess/v001/00_source, then writes each later stage into its own
directory. A second run keeps the snapshot and rewrites only the derived
directories. MinerU output under chem_text_mineru is never opened for writing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prepare_dataset as corpus  # noqa: E402

MINERU_DIR = ROOT / "data" / "chem_dataset" / "chem_text_mineru"
DEFAULT_CONFIG = ROOT / "configs" / "preprocess_v001.yaml"

UNDECODED_RE = re.compile(r"<!--\s*page\s+(\d+)\s+was not decoded\s*-->", re.IGNORECASE)
PAGE_NUMBER_RE = re.compile(r"^[\-—–]*\d{1,4}[\-—–]*$")
EMPTY_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(\s*\)")
HYPHEN_RE = re.compile(r"([A-Za-zА-Яа-яЁё])-\n[ \t]*([a-zа-яё])")
SPECIAL_RE = re.compile(r"\$\$|\$|<table\b", re.IGNORECASE)
TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
WORD_RE = re.compile(r"\S+")
CYR_RE = re.compile(r"[А-Яа-яЁё]")
LAT_RE = re.compile(r"[A-Za-z]")
LETTER_RUN_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+")


def assert_not_under(forbidden: Path, path: Path) -> None:
    resolved = path.resolve()
    root = forbidden.resolve()
    if resolved == root or root in resolved.parents:
        raise SystemExit(f"refusing to write into {resolved}")


def write_text(path: Path, text: str, forbidden: list[Path]) -> None:
    for root in forbidden:
        assert_not_under(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_bytes(path: Path, payload: bytes, forbidden: list[Path]) -> None:
    for root in forbidden:
        assert_not_under(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_config(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data.get("books"):
        raise SystemExit(f"{path} has no books")
    return data


def normalize(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    lines = [line.rstrip(" \t") for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def is_inline_math(body: str) -> bool:
    if "\n\n" in body or len(body) > 400:
        return False
    if any(mark in body for mark in "\\^_{}="):
        return True
    return " " not in body and "\n" not in body and 0 < len(body) <= 40


def split_protected(text: str) -> list[tuple[str, str]]:
    """Keep display math, inline math, and HTML tables as opaque spans."""
    spans: list[tuple[str, str]] = []
    pos = 0
    for match in SPECIAL_RE.finditer(text):
        start = match.start()
        if start < pos:
            continue
        token = match.group(0)
        if token == "$$":
            end_rel = text.find("$$", start + 2)
            if end_rel < 0:
                continue
            end = end_rel + 2
            kind = "math"
        elif token == "$":
            end_rel = text.find("$", start + 1)
            if end_rel < 0 or not is_inline_math(text[start + 1 : end_rel]):
                continue
            end = end_rel + 1
            kind = "math"
        else:
            close = re.search(r"</table>", text[start:], flags=re.IGNORECASE)
            if close is None:
                continue
            end = start + close.end()
            kind = "table"
        if start > pos:
            spans.append(("text", text[pos:start]))
        spans.append((kind, text[start:end]))
        pos = end
    if pos < len(text):
        spans.append(("text", text[pos:]))
    if not spans:
        spans.append(("text", text))
    return spans


def map_text(text: str, fn) -> tuple[str, int]:
    pieces: list[str] = []
    changed = 0
    for kind, chunk in split_protected(text):
        if kind != "text":
            pieces.append(chunk)
            continue
        updated, count = fn(chunk)
        changed += count
        pieces.append(updated)
    return "".join(pieces), changed


def dehyphenate_span(chunk: str) -> tuple[str, int]:
    removed = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal removed
        removed += len(match.group(0)) - 2
        return match.group(1) + match.group(2)

    updated = chunk
    while True:
        updated, count = HYPHEN_RE.subn(repl, updated)
        if count == 0:
            break
    return updated, removed


def strip_images_span(chunk: str) -> tuple[str, int]:
    removed = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal removed
        removed += len(match.group(0))
        return ""

    return EMPTY_IMAGE_RE.sub(repl, chunk), removed


def tidy_span(chunk: str) -> tuple[str, int]:
    updated = re.sub(r"[ \t]+\n", "\n", chunk)
    updated = re.sub(r"\n{3,}", "\n\n", updated)
    return updated, len(chunk) - len(updated)


def header_key(block: str) -> str:
    plain = TAG_RE.sub(" ", block)
    plain = re.sub(r"\s+", " ", plain).strip().casefold()
    return unicodedata.normalize("NFC", plain)


def paragraphs(text: str) -> list[str]:
    return [block for block in re.split(r"\n{2,}", text) if block.strip()]


def join_paragraphs(blocks: list[str]) -> str:
    if not blocks:
        return "\n"
    return "\n\n".join(blocks).strip() + "\n"


def is_page_number(key: str) -> bool:
    return bool(PAGE_NUMBER_RE.match(key))


def is_footnote(block: str) -> bool:
    lowered = block.casefold()
    return "©" in block or "docvortex" in lowered or "page_footnote" in lowered or "<small" in lowered


def mask_protected(text: str) -> tuple[str, list[str], str]:
    """Replace math and tables with tokens so later paragraph edits cannot change them."""
    spans = split_protected(text)
    saved = [chunk for kind, chunk in spans if kind != "text"]
    serial = 0
    while True:
        token_base = f"@@PROT{serial:04d}@"
        if all(token_base not in chunk for kind, chunk in spans if kind == "text"):
            break
        serial += 1
    pieces: list[str] = []
    index = 0
    for kind, chunk in spans:
        if kind == "text":
            pieces.append(chunk)
            continue
        pieces.append(f"{token_base}{index:05d}@")
        index += 1
    return "".join(pieces), saved, token_base


def unmask(text: str, saved: list[str], token_base: str) -> str:
    for index, chunk in enumerate(saved):
        text = text.replace(f"{token_base}{index:05d}@", chunk, 1)
    return text


def remove_repeated_headers(text: str, min_repeats: int, max_chars: int) -> tuple[str, dict]:
    """Drop page numbers and footnotes that repeat. Repeated catalog labels stay."""
    text, saved, token_base = mask_protected(text)
    blocks = paragraphs(text)
    keys = [header_key(block) for block in blocks]
    counts: dict[str, int] = defaultdict(int)
    for key in keys:
        if key:
            counts[key] += 1
    kept: list[str] = []
    removed_chars = 0
    removed_count = 0
    examples: list[str] = []
    for block, key in zip(blocks, keys):
        protected = token_base in block
        page = bool(key) and is_page_number(key)
        footnote = (
            bool(key)
            and len(key) <= max_chars
            and counts[key] >= min_repeats
            and is_footnote(block)
        )
        if not protected and (page or footnote):
            removed_chars += len(block)
            removed_count += 1
            if key not in examples and len(examples) < 5:
                examples.append(key)
            continue
        kept.append(block)
    stats = {
        "header_chars": removed_chars,
        "header_count": removed_count,
        "header_examples": examples,
    }
    return unmask(join_paragraphs(kept), saved, token_base), stats


def meaningful_ratio(text: str) -> float:
    """Share of letters and digits. Punctuation-only OCR noise scores low; formulas and indexes do not."""
    meaningful = 0
    other = 0
    for char in text:
        if char.isspace():
            continue
        if unicodedata.category(char).startswith("L") or char.isdigit():
            meaningful += 1
        else:
            other += 1
    total = meaningful + other
    if total == 0:
        return 0.0
    return meaningful / total


def keep_structured_block(block: str) -> bool:
    if mostly_math_or_table(block):
        return True
    if "$$" in block or block.count("|") >= 2:
        return True
    formulas, _unmatched = extract_formulas(block)
    return bool(formulas)


def mostly_math_or_table(block: str) -> bool:
    if "<table" in block.lower():
        return True
    text_bits: list[str] = []
    has_math = False
    for kind, chunk in split_protected(block):
        if kind == "math":
            has_math = True
        elif kind == "text":
            text_bits.append(chunk)
    prose = "".join(text_bits)
    if has_math and not re.search(r"[A-Za-zА-Яа-яЁё]", prose):
        return True
    return False


def filter_paragraphs(
    text: str,
    garbage_min_chars: int,
    garbage_max_letter_ratio: float,
    dedup_min_chars: int,
) -> tuple[str, dict]:
    undecoded_chars = 0
    undecoded_count = 0

    def drop_comment(match: re.Match[str]) -> str:
        nonlocal undecoded_chars, undecoded_count
        undecoded_chars += len(match.group(0))
        undecoded_count += 1
        return ""

    masked, saved, token_base = mask_protected(text)
    stripped = UNDECODED_RE.sub(drop_comment, masked)
    blocks = paragraphs(stripped)
    garbage_chars = 0
    garbage_count = 0
    garbage_examples: list[str] = []
    survivors: list[str] = []
    for block in blocks:
        if token_base in block or keep_structured_block(block):
            survivors.append(block)
            continue
        if len(block.strip()) >= garbage_min_chars and meaningful_ratio(block) < garbage_max_letter_ratio:
            garbage_chars += len(block)
            garbage_count += 1
            if len(garbage_examples) < 3:
                garbage_examples.append(re.sub(r"\s+", " ", block).strip()[:180])
            continue
        survivors.append(block)

    seen: set[str] = set()
    dedup_chars = 0
    dedup_count = 0
    dedup_examples: list[str] = []
    kept: list[str] = []
    for block in survivors:
        restored = unmask(block, saved, token_base) if token_base in block else block
        key = header_key(restored)
        if len(key) >= dedup_min_chars and key in seen:
            dedup_chars += len(restored)
            dedup_count += 1
            if len(dedup_examples) < 3:
                dedup_examples.append(key[:180])
            continue
        if len(key) >= dedup_min_chars:
            seen.add(key)
        kept.append(block)
    stats = {
        "undecoded_chars": undecoded_chars,
        "undecoded_count": undecoded_count,
        "garbage_chars": garbage_chars,
        "garbage_count": garbage_count,
        "garbage_examples": garbage_examples,
        "dedup_chars": dedup_chars,
        "dedup_count": dedup_count,
        "dedup_examples": dedup_examples,
    }
    return unmask(join_paragraphs(kept), saved, token_base), stats


def clean_text(text: str, min_repeats: int, max_chars: int) -> tuple[str, dict]:
    dehyphenated, dehyphen_chars = map_text(text, dehyphenate_span)
    image_count = 0
    for kind, chunk in split_protected(dehyphenated):
        if kind == "text":
            image_count += len(EMPTY_IMAGE_RE.findall(chunk))
    without_images, image_chars = map_text(dehyphenated, strip_images_span)
    tidied, _tidy_chars = map_text(without_images, tidy_span)
    without_headers, header_stats = remove_repeated_headers(tidied, min_repeats, max_chars)
    stats = {
        "dehyphen_chars": dehyphen_chars,
        "image_chars": image_chars,
        "image_count": image_count,
        **header_stats,
    }
    return without_headers, stats


def process_document(raw: str, config: dict) -> dict:
    normalized = normalize(raw)
    cleaned, clean_stats = clean_text(
        normalized,
        int(config["header_min_repeats"]),
        int(config["header_max_chars"]),
    )
    filtered, filter_stats = filter_paragraphs(
        cleaned,
        int(config["garbage_min_chars"]),
        float(config["garbage_max_letter_ratio"]),
        int(config["dedup_min_chars"]),
    )
    return {
        "normalized": normalized,
        "cleaned": cleaned,
        "filtered": filtered,
        "clean": clean_stats,
        "filter": filter_stats,
    }


def extract_formulas(text: str) -> tuple[list[tuple[str, str]], int]:
    found: list[tuple[str, str]] = []
    pos = 0
    consumed = 0
    for match in re.finditer(r"\$\$|\$", text):
        start = match.start()
        if start < pos:
            continue
        if match.group(0) == "$$":
            end_rel = text.find("$$", start + 2)
            if end_rel < 0:
                continue
            end = end_rel + 2
            found.append(("display", text[start + 2 : end_rel]))
        else:
            end_rel = text.find("$", start + 1)
            if end_rel < 0 or not is_inline_math(text[start + 1 : end_rel]):
                continue
            end = end_rel + 1
            found.append(("inline", text[start + 1 : end_rel]))
        consumed += text[start:end].count("$")
        pos = end
    unmatched = text.count("$") - consumed
    return found, unmatched


def extract_tables(text: str) -> list[str]:
    return [match.group(0) for match in TABLE_RE.finditer(text)]


def mark_kept(source: list, filtered: list) -> tuple[list[bool], bool]:
    marks: list[bool] = []
    index = 0
    for item in source:
        if index < len(filtered) and filtered[index] == item:
            marks.append(True)
            index += 1
        else:
            marks.append(False)
    return marks, index == len(filtered)


def table_shape(table: str) -> tuple[int, int]:
    rows = len(re.findall(r"<tr\b", table, flags=re.IGNORECASE))
    cells = len(re.findall(r"<t[dh]\b", table, flags=re.IGNORECASE))
    return rows, cells


def mixed_script_word(word: str, min_letters: int) -> bool:
    for part in LETTER_RUN_RE.findall(word):
        if len(part) < min_letters:
            continue
        if CYR_RE.search(part) and LAT_RE.search(part):
            return True
    return False


def ocr_signals(text: str, config: dict) -> dict:
    min_letters = int(config["homoglyph_min_letters"])
    garbage_min = int(config["garbage_min_chars"])
    garbage_ratio = float(config["garbage_max_letter_ratio"])
    window_chars = int(config["page_window_chars"])
    blocks = paragraphs(text)
    bad_paragraphs = 0
    homoglyph_paragraphs = 0
    replacement_paragraphs = 0
    low_letter_paragraphs = 0
    for block in blocks:
        homoglyph = any(mixed_script_word(word, min_letters) for word in WORD_RE.findall(block))
        replacement = "\ufffd" in block or "□" in block
        low_letter = (
            not keep_structured_block(block)
            and len(block.strip()) >= garbage_min
            and meaningful_ratio(block) < garbage_ratio
        )
        undecoded = "was not decoded" in block
        if homoglyph:
            homoglyph_paragraphs += 1
        if replacement:
            replacement_paragraphs += 1
        if low_letter:
            low_letter_paragraphs += 1
        if homoglyph or replacement or low_letter or undecoded:
            bad_paragraphs += 1

    windows: list[str] = []
    current: list[str] = []
    current_len = 0
    for block in blocks:
        if current and current_len + len(block) > window_chars:
            windows.append("\n\n".join(current))
            current = [block]
            current_len = len(block)
        else:
            current.append(block)
            current_len += len(block)
    if current:
        windows.append("\n\n".join(current))

    bad_windows = 0
    for window in windows:
        words = WORD_RE.findall(window)
        mixed = sum(1 for word in words if mixed_script_word(word, min_letters))
        ratio = mixed / len(words) if words else 0.0
        broken = (
            "was not decoded" in window
            or "\ufffd" in window
            or "□" in window
            or (
                not keep_structured_block(window)
                and len(window) >= garbage_min
                and meaningful_ratio(window) < garbage_ratio
            )
            or (mixed >= 2 and ratio > 0.05)
        )
        if broken:
            bad_windows += 1

    paragraph_count = len(blocks)
    window_count = len(windows)
    return {
        "paragraphs": paragraph_count,
        "bad_paragraphs": bad_paragraphs,
        "bad_paragraph_share": bad_paragraphs / paragraph_count if paragraph_count else 0.0,
        "homoglyph_paragraphs": homoglyph_paragraphs,
        "replacement_paragraphs": replacement_paragraphs,
        "low_letter_paragraphs": low_letter_paragraphs,
        "windows": window_count,
        "bad_windows": bad_windows,
        "bad_window_share": bad_windows / window_count if window_count else 0.0,
        "estimated_pages": window_count,
    }


def count_tokens(tokenizer, text: str) -> int:
    if not text.strip():
        return 0
    return len(tokenizer.encode(text).ids)


def snapshot_books(config: dict, source_dir: Path, snap_dir: Path, forbidden: list[Path]) -> list[dict]:
    records = []
    for book in config["books"]:
        name = book["file"]
        src = source_dir / name
        if not src.is_file():
            raise SystemExit(f"missing source book: {src}")
        dest = snap_dir / name
        if not dest.is_file():
            payload = src.read_bytes()
            write_bytes(dest, payload, forbidden)
        else:
            payload = dest.read_bytes()
        records.append(
            {
                "file": name,
                "reason": book["reason"],
                "sha256": sha256_bytes(payload),
                "bytes": len(payload),
            }
        )
    return records


def context_around(text: str, needle: str, window: int = 90) -> str:
    index = text.find(needle)
    if index < 0:
        return ""
    start = max(0, index - window)
    end = min(len(text), index + len(needle) + window)
    snippet = text[start:end].replace("\n", " ")
    return re.sub(r"\s+", " ", snippet).strip()


def formula_chunk(kind: str, body: str) -> str:
    if kind == "display":
        return "$$" + body + "$$"
    return "$" + body + "$"


def take_round_robin(groups: list[list], limit: int) -> list:
    chosen = []
    index = 0
    while len(chosen) < limit:
        progressed = False
        for group in groups:
            if index < len(group):
                chosen.append(group[index])
                progressed = True
                if len(chosen) >= limit:
                    break
        if not progressed:
            break
        index += 1
    return chosen


def render_formula_samples(samples: list[dict]) -> str:
    lines = [
        "# Фрагменты формул",
        "",
        "Фрагменты взяты из нормализованного Markdown по кругу из всех книг.",
        "Строка «после» говорит, остался ли тот же блок в отфильтрованном тексте.",
        "",
    ]
    for number, sample in enumerate(samples, 1):
        body = sample["body"].strip()
        if len(body) > 400:
            body = body[:400] + "…"
        lines.append(f"## {number}. {sample['file']}")
        lines.append("")
        lines.append(f"Вид: {sample['kind']}. После фильтра: {'сохранён' if sample['kept'] else 'снят'}.")
        lines.append("")
        lines.append("```")
        lines.append(body if body else "(пустой блок)")
        lines.append("```")
        if sample["context"]:
            lines.append("")
            lines.append(f"Контекст: {sample['context']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_table_samples(samples: list[dict]) -> str:
    lines = [
        "# Фрагменты таблиц",
        "",
        "Таблица сохранена, если её HTML после фильтра совпал с нормализованным.",
        "",
    ]
    for number, sample in enumerate(samples, 1):
        html = sample["html"]
        shown = html if len(html) <= 1200 else html[:1200] + "…"
        lines.append(f"## {number}. {sample['file']}")
        lines.append("")
        lines.append(
            f"Строк: {sample['rows']}, ячеек: {sample['cells']}. "
            f"После фильтра: {'сохранена' if sample['kept'] else 'снята'}."
        )
        lines.append("")
        lines.append("```html")
        lines.append(shown)
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def chunk_id(text: str) -> str:
    folded = unicodedata.normalize("NFC", re.sub(r"\s+", " ", text).strip().casefold())
    return sha256_bytes(folded.encode("utf-8"))


def cross_book_duplicates(documents: list[tuple[str, str]], chunk_min_chars: int) -> dict:
    exact_groups: dict[str, list[dict]] = defaultdict(list)
    chunks: list[dict] = []
    for book, text in documents:
        for block in paragraphs(text):
            if len(block) < chunk_min_chars:
                continue
            digest = chunk_id(block)
            item = {"book": book, "text": block, "sha": digest}
            chunks.append(item)
            exact_groups[digest].append(item)

    exact_records = []
    for digest, group in exact_groups.items():
        books = sorted({item["book"] for item in group})
        if len(books) < 2:
            continue
        snippet = re.sub(r"\s+", " ", group[0]["text"]).strip()[:180]
        exact_records.append(
            {
                "kind": "exact_chunk",
                "books": books,
                "copies": len(group),
                "chars": len(group[0]["text"]),
                "sha256": digest,
                "snippet": snippet,
            }
        )
    exact_records.sort(key=lambda record: (-record["copies"], -record["chars"]))

    bitmap = corpus.build_space_bitmap()
    coefficients_a, coefficients_b = corpus.minhash_coefficients(corpus.SEED)
    near_records = []
    truncated = False
    closest = {"jaccard": -1.0, "left": "", "right": ""}
    if chunks:
        signatures = np.vstack(
            [
                corpus.signature_of_text(item["text"], bitmap, coefficients_a, coefficients_b)
                for item in chunks
            ]
        )
        bands = corpus.band_hashes(signatures)
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, item in enumerate(chunks):
            for band in range(corpus.N_BANDS):
                buckets[(band, int(bands[band, index]))].append(index)
        candidates: set[tuple[int, int]] = set()
        for members in buckets.values():
            by_book: dict[str, list[int]] = defaultdict(list)
            for index in members:
                by_book[chunks[index]["book"]].append(index)
            names = list(by_book)
            if len(names) < 2:
                continue
            for left_name_i in range(len(names)):
                for right_name_i in range(left_name_i + 1, len(names)):
                    lefts = by_book[names[left_name_i]]
                    rights = by_book[names[right_name_i]]
                    if len(lefts) * len(rights) > 4000:
                        truncated = True
                        rights = rights[:20]
                        lefts = lefts[:20]
                    for left in lefts:
                        for right in rights:
                            pair = (left, right) if left < right else (right, left)
                            if chunks[pair[0]]["sha"] == chunks[pair[1]]["sha"]:
                                continue
                            candidates.add(pair)
                            if len(candidates) > 250_000:
                                truncated = True
                                break
                        if len(candidates) > 250_000:
                            break
                if len(candidates) > 250_000:
                    break
            if len(candidates) > 250_000:
                break
            if len(candidates) > 250_000:
                break
        shingles: dict[int, np.ndarray] = {}
        for left, right in candidates:
            for index in (left, right):
                if index not in shingles:
                    shingles[index] = corpus.shingles_of_text(chunks[index]["text"], bitmap)
            score, ok = corpus.jaccard_pair(shingles[left], shingles[right])
            if score > closest["jaccard"]:
                closest = {
                    "jaccard": score,
                    "left": chunks[left]["book"],
                    "right": chunks[right]["book"],
                }
            if not ok:
                continue
            near_records.append(
                {
                    "kind": "near_chunk",
                    "left": chunks[left]["book"],
                    "right": chunks[right]["book"],
                    "jaccard": round(float(score), 4),
                    "snippet": re.sub(r"\s+", " ", chunks[left]["text"]).strip()[:180],
                }
            )
    near_records.sort(key=lambda record: -record["jaccard"])

    document_pairs = []
    doc_shingles = [
        (book, corpus.shingles_of_text(text, bitmap))
        for book, text in documents
    ]
    for left in range(len(doc_shingles)):
        for right in range(left + 1, len(doc_shingles)):
            score, ok = corpus.jaccard_pair(doc_shingles[left][1], doc_shingles[right][1])
            document_pairs.append(
                {
                    "kind": "document",
                    "left": doc_shingles[left][0],
                    "right": doc_shingles[right][0],
                    "jaccard": round(float(score), 4),
                    "near": bool(ok),
                }
            )
    document_pairs.sort(key=lambda record: -record["jaccard"])
    return {
        "exact": exact_records,
        "near": near_records,
        "documents": document_pairs,
        "truncated": truncated,
        "closest_candidate": closest,
        "chunks": len(chunks),
    }


def percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def short_name(name: str) -> str:
    stem = name[:-3] if name.endswith(".md") else name
    if len(stem) <= 48:
        return stem
    return stem[:45] + "…"


def render_summary(config: dict, books: list[dict], dups: dict) -> str:
    norm_chars = sum(book["chars_normalized"] for book in books)
    filt_chars = sum(book["chars_filtered"] for book in books)
    norm_tokens = sum(book["tokens_normalized"] for book in books)
    filt_tokens = sum(book["tokens_filtered"] for book in books)
    content_loss = sum(book["content_loss_chars"] for book in books)
    dehyphen = sum(book["dehyphen_chars"] for book in books)
    normalize_delta = sum(book["normalize_delta_chars"] for book in books)
    file_delta = norm_chars - filt_chars
    reasons = {
        "колонтитулы и повторы сносок": sum(book["header_chars"] for book in books),
        "пустые картинки": sum(book["image_chars"] for book in books),
        "непрочитанные страницы": sum(book["undecoded_chars"] for book in books),
        "абзацы почти без букв и цифр": sum(book["garbage_chars"] for book in books),
        "точные копии абзацев внутри книги": sum(book["dedup_chars"] for book in books),
    }
    formula_src = sum(book["formulas_normalized"] for book in books)
    formula_kept = sum(book["formulas_kept"] for book in books)
    formula_removed = sum(book["formulas_removed"] for book in books)
    formula_empty = sum(book["formulas_empty"] for book in books)
    formula_unmatched = sum(book["formulas_unmatched"] for book in books)
    formula_changed = sum(1 for book in books if not book["formulas_intact"])
    table_src = sum(book["tables_normalized"] for book in books)
    table_kept = sum(book["tables_kept"] for book in books)
    table_removed = sum(book["tables_removed"] for book in books)
    table_changed = sum(1 for book in books if not book["tables_intact"])
    table_rows = sum(book["table_rows"] for book in books)
    table_cells = sum(book["table_cells"] for book in books)
    undecoded = sum(book["undecoded_count"] for book in books)
    paragraphs = sum(book["ocr_paragraphs"] for book in books)
    bad_paragraphs = sum(book["ocr_bad_paragraphs"] for book in books)
    windows = sum(book["ocr_windows"] for book in books)
    bad_windows = sum(book["ocr_bad_windows"] for book in books)
    attributed = sum(reasons.values())
    residual = file_delta - attributed - dehyphen

    lines = [
        "# Пилот preprocessing v001",
        "",
        "Исходные Markdown в `data/chem_dataset/chem_text_mineru` не изменялись.",
        "Снимок 16 книг лежит в `00_source`. Нормализация, очистка и фильтр — в отдельных каталогах.",
        "Токены посчитаны словарём `tokenizers/fineweb2_rus_v001/chosen` (byte-level BPE, 16 000).",
        "",
        "## Токены",
        "",
        f"- До очистки, на нормализованном тексте: {norm_tokens} токенов, {norm_chars} символов, "
        f"{(norm_chars / norm_tokens) if norm_tokens else 0:.2f} символа на токен.",
        f"- После фильтра: {filt_tokens} токенов, {filt_chars} символов, "
        f"{(filt_chars / filt_tokens) if filt_tokens else 0:.2f} символа на токен.",
        "",
        "## Потеря текста",
        "",
        "Потеря содержания считается от нормализованного текста к отфильтрованному и не включает склейку переносов.",
        f"Снято как содержание: {content_loss} символов, {percent(content_loss / norm_chars) if norm_chars else '0.0%'} от нормализованного текста.",
        f"Склейка переносов убрала {dehyphen} символов (дефис и перевод строки у склеенного слова).",
        f"Нормализация пробелов и Unicode относительно снимка изменила длину на {normalize_delta} символов.",
        f"Разница длины файлов между нормализацией и фильтром: {file_delta} символов. "
        f"Из них на пустые строки после удалений приходится {residual}.",
        "",
        "По причинам:",
    ]
    for reason, chars in reasons.items():
        share = chars / norm_chars if norm_chars else 0.0
        lines.append(f"- {reason}: {chars} символов ({percent(share)})")
    lines.extend(["", "Примеры снятого:"])
    shown = 0
    for book in books:
        for example in book["header_examples"]:
            if shown >= 6:
                break
            lines.append(f"- колонтитул, {short_name(book['file'])}: {example[:160]}")
            shown += 1
        for example in book["garbage_examples"]:
            if shown >= 10:
                break
            lines.append(f"- почти без букв и цифр, {short_name(book['file'])}: {example}")
            shown += 1
        for example in book["dedup_examples"]:
            if shown >= 14:
                break
            lines.append(f"- точная копия абзаца, {short_name(book['file'])}: {example}")
            shown += 1
    if shown == 0:
        lines.append("- Снятых колонтитулов и мусорных абзацев нет.")

    lines.extend(
        [
            "",
            "## Формулы",
            "",
            "Считаются закрытые блоки `$$...$$` и короткие `$...$` с признаками формулы. Каждый знак `$` в тексте не считается формулой.",
            f"В нормализованном тексте: {formula_src} блоков, из них пустых {formula_empty}.",
            f"После фильтра на месте остались {formula_kept}, сняты вместе с абзацем {formula_removed}.",
            f"Книг, где последовательность формул изменилась не только за счёт удаления: {formula_changed}.",
            f"Знаков `$`, не вошедших в формулу: {formula_unmatched}.",
            "Химическая верность формул по этому отчёту не проверяется. Для просмотра есть `formula_samples.md`.",
            "",
            "## Таблицы",
            "",
            f"В нормализованном тексте: {table_src} таблиц, {table_rows} строк, {table_cells} ячеек.",
            f"HTML совпал после фильтра у {table_kept} таблиц, снято {table_removed}.",
            f"Книг, где последовательность таблиц изменилась не только за счёт удаления: {table_changed}.",
            "Десять таблиц лежат в `table_samples.md`.",
            "",
            "## Ошибки OCR",
            "",
            "Облачный Markdown не хранит границы страниц. Ниже три отдельных числа.",
            f"- Явные непрочитанные страницы `<!-- page N was not decoded -->`: {undecoded}. "
            f"Это {percent(undecoded / windows) if windows else '0.0%'} от оценочного числа окон.",
            f"- Абзацы с признаком OCR: {bad_paragraphs} из {paragraphs} "
            f"({percent(bad_paragraphs / paragraphs) if paragraphs else '0.0%'}). "
            "Признак — замена символа, почти нет букв и цифр, или буква из латиницы и кириллицы в одном сплошном фрагменте слова. "
            "Абзац с формулой или таблицей этим правилом не помечается. Дефисные формулы вроде `C-C-связь` не считаются гомоглифом.",
            f"  Из них с гомоглифом: {sum(book['ocr_homoglyph_paragraphs'] for book in books)}, "
            f"с символом замены: {sum(book['ocr_replacement_paragraphs'] for book in books)}, "
            f"почти без букв и цифр: {sum(book['ocr_low_letter_paragraphs'] for book in books)}.",
            f"- Оценка доли страниц: окна примерно по {config['page_window_chars']} символов. "
            f"Плохих окон {bad_windows} из {windows} ({percent(bad_windows / windows) if windows else '0.0%'}). "
            "Окно плохое, если в нём есть явный пропуск, символ замены, почти нет букв и цифр или доля гомоглифов выше 5% при хотя бы двух таких словах. "
            "Это оценка, не номер страницы MinerU.",
            "",
            "## Дубли",
            "",
            f"Точных копий абзацев внутри книги снято: {sum(book['dedup_count'] for book in books)}.",
            f"Длинных абзацев (от {config['chunk_min_chars']} символов) в отфильтрованном тексте: {dups['chunks']}.",
            f"Групп точных абзацев, встретившихся в двух и более книгах: {len(dups['exact'])}.",
            f"Пар абзацев из разных книг с Jaccard не ниже 4/5: {len(dups['near'])}.",
            f"Пар книг целиком с Jaccard не ниже 4/5: {sum(1 for pair in dups['documents'] if pair['near'])}.",
        ]
    )
    if dups["documents"]:
        top = dups["documents"][0]
        lines.append(
            f"Ближайшие книги целиком: {short_name(top['left'])} и {short_name(top['right'])}, Jaccard {top['jaccard']:.2f}."
        )
    if dups["truncated"]:
        lines.append("Поиск близких абзацев упёрся в предохранитель на размере корзины MinHash. Число пар может быть неполным.")
    if dups["exact"]:
        lines.append("")
        lines.append("Примеры точных межкнижных абзацев:")
        for record in dups["exact"][:8]:
            names = ", ".join(short_name(name) for name in record["books"])
            lines.append(f"- {record['copies']} копий, {names}: {record['snippet']}")
    if dups["near"]:
        lines.append("")
        lines.append("Примеры близких абзацев:")
        for record in dups["near"][:8]:
            lines.append(
                f"- {record['jaccard']:.2f}, {short_name(record['left'])} / {short_name(record['right'])}: {record['snippet']}"
            )
    lines.extend(["", "## Книги", ""])
    for book in books:
        loss = book["content_loss_chars"] / book["chars_normalized"] if book["chars_normalized"] else 0.0
        lines.append(
            f"- {book['file']}: {book['tokens_filtered']} токенов после фильтра, "
            f"потеря содержания {percent(loss)}, "
            f"формулы {book['formulas_kept']}/{book['formulas_normalized']}, "
            f"таблицы {book['tables_kept']}/{book['tables_normalized']}, "
            f"оценка плохих окон {percent(book['ocr_bad_window_share'])}. "
            f"{book['reason']}"
        )
    lines.append("")
    return "\n".join(lines)


def run(config_path: Path) -> None:
    config = load_config(config_path)
    source_dir = ROOT / config["source_dir"]
    output_dir = ROOT / config["output_dir"]
    forbidden = [source_dir, MINERU_DIR]
    assert_not_under(source_dir, output_dir)
    if source_dir.resolve() != MINERU_DIR.resolve():
        assert_not_under(MINERU_DIR, output_dir)

    tokenizer_path = ROOT / config["tokenizer"]
    tokenizer = corpus.Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != corpus.VOCAB_SIZE:
        raise SystemExit(f"vocab is {tokenizer.get_vocab_size()}, expected {corpus.VOCAB_SIZE}")

    snap_dir = output_dir / "00_source"
    records = snapshot_books(config, source_dir, snap_dir, forbidden)
    try:
        config_label = str(config_path.resolve().relative_to(ROOT))
    except ValueError:
        config_label = str(config_path)
    selection = {
        "version": config["version"],
        "source_dir": config["source_dir"],
        "output_dir": config["output_dir"],
        "config": config_label,
        "books": records,
    }
    write_text(
        output_dir / "selection.json",
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        forbidden,
    )

    stage_names = ("01_normalized", "02_cleaned", "03_filtered")
    stage_dirs = {name: output_dir / name for name in stage_names}
    book_reports = []
    formula_groups: list[list[dict]] = []
    table_groups: list[list[dict]] = []
    filtered_docs: list[tuple[str, str]] = []

    for record in records:
        name = record["file"]
        print(f"process {name}", flush=True)
        raw = (snap_dir / name).read_text(encoding="utf-8")
        result = process_document(raw, config)
        write_text(stage_dirs["01_normalized"] / name, result["normalized"], forbidden)
        write_text(stage_dirs["02_cleaned"] / name, result["cleaned"], forbidden)
        write_text(stage_dirs["03_filtered"] / name, result["filtered"], forbidden)

        clean_stats = result["clean"]
        filter_stats = result["filter"]
        content_loss = (
            clean_stats["header_chars"]
            + clean_stats["image_chars"]
            + filter_stats["undecoded_chars"]
            + filter_stats["garbage_chars"]
            + filter_stats["dedup_chars"]
        )
        source_formulas, unmatched = extract_formulas(result["normalized"])
        filtered_formulas, _filtered_unmatched = extract_formulas(result["filtered"])
        formula_marks, formulas_intact = mark_kept(source_formulas, filtered_formulas)
        source_tables = extract_tables(result["normalized"])
        filtered_tables = extract_tables(result["filtered"])
        table_marks, tables_intact = mark_kept(source_tables, filtered_tables)
        rows = 0
        cells = 0
        for table in source_tables:
            row_count, cell_count = table_shape(table)
            rows += row_count
            cells += cell_count
        ocr = ocr_signals(result["normalized"], config)
        tokens_normalized = count_tokens(tokenizer, result["normalized"])
        tokens_filtered = count_tokens(tokenizer, result["filtered"])
        report = {
            "file": name,
            "reason": record["reason"],
            "sha256": record["sha256"],
            "bytes": record["bytes"],
            "chars_source": len(raw),
            "chars_normalized": len(result["normalized"]),
            "chars_cleaned": len(result["cleaned"]),
            "chars_filtered": len(result["filtered"]),
            "normalize_delta_chars": len(raw) - len(result["normalized"]),
            "tokens_normalized": tokens_normalized,
            "tokens_filtered": tokens_filtered,
            "chars_per_token_normalized": (len(result["normalized"]) / tokens_normalized) if tokens_normalized else 0.0,
            "chars_per_token_filtered": (len(result["filtered"]) / tokens_filtered) if tokens_filtered else 0.0,
            "dehyphen_chars": clean_stats["dehyphen_chars"],
            "image_chars": clean_stats["image_chars"],
            "image_count": clean_stats["image_count"],
            "header_chars": clean_stats["header_chars"],
            "header_count": clean_stats["header_count"],
            "header_examples": clean_stats["header_examples"],
            "undecoded_chars": filter_stats["undecoded_chars"],
            "undecoded_count": filter_stats["undecoded_count"],
            "garbage_chars": filter_stats["garbage_chars"],
            "garbage_count": filter_stats["garbage_count"],
            "garbage_examples": filter_stats["garbage_examples"],
            "dedup_chars": filter_stats["dedup_chars"],
            "dedup_count": filter_stats["dedup_count"],
            "dedup_examples": filter_stats["dedup_examples"],
            "content_loss_chars": content_loss,
            "formulas_normalized": len(source_formulas),
            "formulas_filtered": len(filtered_formulas),
            "formulas_kept": sum(formula_marks),
            "formulas_removed": len(formula_marks) - sum(formula_marks),
            "formulas_empty": sum(1 for _kind, body in source_formulas if not body.strip()),
            "formulas_unmatched": unmatched,
            "formulas_intact": formulas_intact,
            "formulas_display": sum(1 for kind, _body in source_formulas if kind == "display"),
            "formulas_inline": sum(1 for kind, _body in source_formulas if kind == "inline"),
            "tables_normalized": len(source_tables),
            "tables_filtered": len(filtered_tables),
            "tables_kept": sum(table_marks),
            "tables_removed": len(table_marks) - sum(table_marks),
            "tables_intact": tables_intact,
            "table_rows": rows,
            "table_cells": cells,
            "ocr_paragraphs": ocr["paragraphs"],
            "ocr_bad_paragraphs": ocr["bad_paragraphs"],
            "ocr_bad_paragraph_share": ocr["bad_paragraph_share"],
            "ocr_homoglyph_paragraphs": ocr["homoglyph_paragraphs"],
            "ocr_replacement_paragraphs": ocr["replacement_paragraphs"],
            "ocr_low_letter_paragraphs": ocr["low_letter_paragraphs"],
            "ocr_windows": ocr["windows"],
            "ocr_bad_windows": ocr["bad_windows"],
            "ocr_bad_window_share": ocr["bad_window_share"],
        }
        book_reports.append(report)
        filtered_docs.append((name, result["filtered"]))

        formula_group = []
        for (kind, body), kept in zip(source_formulas, formula_marks):
            chunk = formula_chunk(kind, body)
            formula_group.append(
                {
                    "file": name,
                    "kind": "блочная" if kind == "display" else "строчная",
                    "body": body,
                    "kept": kept,
                    "context": context_around(result["normalized"], chunk),
                }
            )
        formula_groups.append(formula_group)
        table_group = []
        for table, kept in zip(source_tables, table_marks):
            row_count, cell_count = table_shape(table)
            table_group.append(
                {
                    "file": name,
                    "html": table,
                    "rows": row_count,
                    "cells": cell_count,
                    "kept": kept,
                }
            )
        table_groups.append(table_group)

    print("duplicates", flush=True)
    dups = cross_book_duplicates(filtered_docs, int(config["chunk_min_chars"]))
    report_dir = output_dir / "report"
    summary = render_summary(config, book_reports, dups)
    write_text(report_dir / "summary.md", summary, forbidden)
    jsonl = "".join(json.dumps(book, ensure_ascii=False) + "\n" for book in book_reports)
    write_text(report_dir / "books.jsonl", jsonl, forbidden)
    samples = take_round_robin(formula_groups, int(config["formula_samples"]))
    write_text(report_dir / "formula_samples.md", render_formula_samples(samples), forbidden)
    table_samples = take_round_robin(table_groups, int(config["table_samples"]))
    write_text(report_dir / "table_samples.md", render_table_samples(table_samples), forbidden)
    dup_lines = []
    for record in dups["exact"]:
        dup_lines.append(json.dumps(record, ensure_ascii=False))
    for record in dups["near"]:
        dup_lines.append(json.dumps(record, ensure_ascii=False))
    for record in dups["documents"]:
        if record["near"]:
            dup_lines.append(json.dumps(record, ensure_ascii=False))
    write_text(report_dir / "duplicates.jsonl", ("\n".join(dup_lines) + "\n") if dup_lines else "", forbidden)
    print(f"wrote {report_dir / 'summary.md'}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MinerU preprocessing pilot.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
