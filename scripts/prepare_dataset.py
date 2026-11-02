#!/usr/bin/env python3
"""Freeze the train / validation / test token streams.

The 16k tokenizer is fixed. The default corpus is fineweb2-rus-v001, and its
part hashes must match tokenizers/fineweb2_rus_v001/chosen/choice.json.
--corpus points at another directory with the same layout (manifest.json and
parts/part-*.parquet with columns id and text). That path checks hashes
against its own manifest and does not touch choice.json.

The script splits documents 98/1/1, drops exact copies and near-copies that
would leak into the held-out splits, and writes a uint16 token stream with an
EOS after every kept document.

Outputs, under <corpus>/dataset/ unless --out is set:
  manifest.json
  assignments.parquet
  duplicates.jsonl
  train.txt, validation.txt, test.txt
  tokens/train.bin, tokens/validation.bin, tokens/test.bin

Id lists and token files follow corpus order (part name, then row). The
permutation only chooses the split. A finished manifest is left untouched
unless --refresh is passed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import shutil
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numba
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
FINEWEB_CORPUS = ROOT / "data" / "fineweb2" / "rus_Cyrl" / "v001"
CORPUS_DIR = FINEWEB_CORPUS
PARTS_DIR = CORPUS_DIR / "parts"
CORPUS_MANIFEST = CORPUS_DIR / "manifest.json"
CHOSEN_DIR = ROOT / "tokenizers" / "fineweb2_rus_v001" / "chosen"
BPE_DIR = ROOT / "tokenizers" / "fineweb2_rus_v001" / "bpe-16000"
DEFAULT_OUT = CORPUS_DIR / "dataset"

TRAIN, VAL, TEST = 0, 1, 2
KEPT, EXACT, NEAR = 0, 1, 2
SPLIT_NAME = ("train", "validation", "test")
STATUS_NAME = ("kept", "discarded_exact_dup", "discarded_near_dup")

SEED = 22
VOCAB_SIZE = 16_000
EOS_TOKEN = "<|endoftext|>"
N_DOCS_EXPECTED = 442_930
ROUNDTRIP_DOCS = 2_000

N_PERM = 32
N_BANDS = 8
BAND_ROWS = 4
JACCARD_NUM = 4
JACCARD_DEN = 5

FNV_OFFSET = np.uint64(14695981039346656037)
FNV_PRIME = np.uint64(1099511628211)
SHINGLE_P = np.uint64(0x9E3779B185EBCA87)
MASK64 = (1 << 64) - 1


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def canonical_fingerprint(entries: list[dict]) -> str:
    blob = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(blob.encode("utf-8"))


def quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    keys = [10, 25, 50, 75, 90, 99]
    points = np.percentile(values, keys)
    return {f"p{key}": float(point) for key, point in zip(keys, points)}


def length_stats(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {
            "docs": 0,
            "mean": 0.0,
            "max": 0,
            "frac_gt_512": 0.0,
            "frac_gt_1024": 0.0,
        }
    stats: dict[str, float | int] = {
        "docs": int(values.size),
        "mean": float(values.mean()),
        "max": int(values.max()),
        "frac_gt_512": float(np.mean(values > 512)),
        "frac_gt_1024": float(np.mean(values > 1024)),
    }
    stats.update(quantiles(values))
    return stats


def build_space_bitmap() -> np.ndarray:
    bitmap = np.zeros(0x110000, dtype=np.uint8)
    for codepoint in range(0x110000):
        if chr(codepoint).isspace():
            bitmap[codepoint] = 1
    return bitmap


def minhash_coefficients(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 2**63, size=(2, N_PERM), dtype=np.uint64)
    coefficients_a = np.ascontiguousarray(raw[0] | np.uint64(1))
    coefficients_b = np.ascontiguousarray(raw[1])
    return coefficients_a, coefficients_b


def fnv_word(word: str) -> int:
    hash_value = int(FNV_OFFSET)
    for byte in word.encode("utf-8"):
        hash_value ^= byte
        hash_value = (hash_value * int(FNV_PRIME)) & MASK64
    return hash_value


def python_word_hashes(text: str) -> np.ndarray:
    import re

    words = re.findall(r"\S+", nfc(text).casefold())
    return np.asarray([fnv_word(word) for word in words], dtype=np.uint64)


def python_shingles(word_hash: np.ndarray) -> np.ndarray:
    count = int(word_hash.shape[0])
    prime = int(SHINGLE_P)
    if count == 0:
        return np.empty(0, dtype=np.uint64)
    if count < 5:
        gram = int(word_hash[0])
        for index in range(1, count):
            gram = (gram * prime + int(word_hash[index])) & MASK64
        return np.asarray([gram], dtype=np.uint64)
    grams = np.empty(count - 4, dtype=np.uint64)
    for index in range(count - 4):
        gram = int(word_hash[index])
        for shift in range(1, 5):
            gram = (gram * prime + int(word_hash[index + shift])) & MASK64
        grams[index] = gram
    return grams


@numba.njit
def _codepoint_at(data, index):
    size = data.shape[0]
    first = np.uint32(data[index])
    if first < np.uint32(0x80):
        return first, index + 1
    if first < np.uint32(0xE0) and index + 1 < size:
        codepoint = ((first & np.uint32(0x1F)) << np.uint32(6)) | (np.uint32(data[index + 1]) & np.uint32(0x3F))
        return codepoint, index + 2
    if first < np.uint32(0xF0) and index + 2 < size:
        codepoint = (
            ((first & np.uint32(0x0F)) << np.uint32(12))
            | ((np.uint32(data[index + 1]) & np.uint32(0x3F)) << np.uint32(6))
            | (np.uint32(data[index + 2]) & np.uint32(0x3F))
        )
        return codepoint, index + 3
    if index + 3 < size:
        codepoint = (
            ((first & np.uint32(0x07)) << np.uint32(18))
            | ((np.uint32(data[index + 1]) & np.uint32(0x3F)) << np.uint32(12))
            | ((np.uint32(data[index + 2]) & np.uint32(0x3F)) << np.uint32(6))
            | (np.uint32(data[index + 3]) & np.uint32(0x3F))
        )
        return codepoint, index + 4
    return first, index + 1


@numba.njit
def _is_space(codepoint, space_bitmap) -> bool:
    if codepoint >= np.uint32(space_bitmap.shape[0]):
        return False
    return space_bitmap[codepoint] == np.uint8(1)


@numba.njit
def word_hashes(data, space_bitmap):
    size = data.shape[0]
    count = 0
    index = 0
    in_word = False
    while index < size:
        codepoint, nxt = _codepoint_at(data, index)
        if _is_space(codepoint, space_bitmap):
            in_word = False
        elif not in_word:
            count += 1
            in_word = True
        index = nxt

    hashes = np.empty(count, dtype=np.uint64)
    if count == 0:
        return hashes

    index = 0
    slot = 0
    in_word = False
    hash_value = FNV_OFFSET
    while index < size:
        start = index
        codepoint, nxt = _codepoint_at(data, index)
        if _is_space(codepoint, space_bitmap):
            if in_word:
                hashes[slot] = hash_value
                slot += 1
                in_word = False
        else:
            if not in_word:
                hash_value = FNV_OFFSET
                in_word = True
            for byte_index in range(start, nxt):
                hash_value ^= np.uint64(data[byte_index])
                hash_value *= FNV_PRIME
        index = nxt
    if in_word:
        hashes[slot] = hash_value
        slot += 1
    return hashes[:slot]


@numba.njit
def shingle_hashes(word_hash):
    count = word_hash.shape[0]
    if count == 0:
        return np.empty(0, dtype=np.uint64)
    if count < 5:
        gram = word_hash[0]
        for index in range(1, count):
            gram = gram * SHINGLE_P + word_hash[index]
        out = np.empty(1, dtype=np.uint64)
        out[0] = gram
        return out
    out = np.empty(count - 4, dtype=np.uint64)
    for index in range(count - 4):
        gram = word_hash[index]
        gram = gram * SHINGLE_P + word_hash[index + 1]
        gram = gram * SHINGLE_P + word_hash[index + 2]
        gram = gram * SHINGLE_P + word_hash[index + 3]
        gram = gram * SHINGLE_P + word_hash[index + 4]
        out[index] = gram
    return out


@numba.njit
def minhash_signature(shingles, coefficients_a, coefficients_b):
    width = coefficients_a.shape[0]
    signature = np.empty(width, dtype=np.uint64)
    limit = np.uint64(0xFFFFFFFFFFFFFFFF)
    for perm in range(width):
        best = limit
        if shingles.shape[0] > 0:
            scale = coefficients_a[perm]
            shift = coefficients_b[perm]
            for index in range(shingles.shape[0]):
                mixed = shingles[index] * scale + shift
                if mixed < best:
                    best = mixed
        signature[perm] = best
    return signature


@numba.njit
def band_hashes(signatures):
    count = signatures.shape[0]
    out = np.empty((N_BANDS, count), dtype=np.uint64)
    for doc_index in range(count):
        for band in range(N_BANDS):
            hash_value = FNV_OFFSET
            base = band * BAND_ROWS
            for row in range(BAND_ROWS):
                hash_value ^= signatures[doc_index, base + row]
                hash_value *= FNV_PRIME
            out[band, doc_index] = hash_value
    return out


def folded_bytes(text: str) -> np.ndarray:
    raw = nfc(text).casefold().encode("utf-8")
    return np.frombuffer(raw, dtype=np.uint8).copy()


def shingles_of_text(text: str, space_bitmap: np.ndarray) -> np.ndarray:
    return shingle_hashes(word_hashes(folded_bytes(text), space_bitmap))


def signature_of_text(text: str, space_bitmap: np.ndarray, coefficients_a: np.ndarray, coefficients_b: np.ndarray) -> np.ndarray:
    return minhash_signature(shingles_of_text(text, space_bitmap), coefficients_a, coefficients_b)


def jaccard_pair(left: np.ndarray, right: np.ndarray) -> tuple[float, bool]:
    unique_left = np.unique(left)
    unique_right = np.unique(right)
    i = 0
    j = 0
    inter = 0
    left_n = int(unique_left.size)
    right_n = int(unique_right.size)
    while i < left_n and j < right_n:
        left_value = int(unique_left[i])
        right_value = int(unique_right[j])
        if left_value == right_value:
            inter += 1
            i += 1
            j += 1
        elif left_value < right_value:
            i += 1
        else:
            j += 1
    union = left_n + right_n - inter
    if union == 0:
        return 1.0, True
    return inter / union, inter * JACCARD_DEN >= union * JACCARD_NUM


def assign_splits(n_docs: int, seed: int) -> np.ndarray:
    """First n//100 of the permutation are validation, the next n//100 are test."""
    n_val = n_docs // 100
    n_test = n_docs // 100
    initial = np.full(n_docs, 255, dtype=np.uint8)
    permutation = np.random.default_rng(seed).permutation(n_docs)
    initial[permutation[:n_val]] = VAL
    initial[permutation[n_val : n_val + n_test]] = TEST
    initial[permutation[n_val + n_test :]] = TRAIN
    if np.any(initial == 255):
        raise RuntimeError("Split assignment missed a document.")
    return initial


def apply_exact(sha: np.ndarray, ids: list[str], status: np.ndarray, match_index: np.ndarray) -> None:
    """Keep the lexicographically smallest id in each identical NFC text."""
    void = np.ascontiguousarray(sha).view(np.dtype((np.void, sha.shape[1]))).ravel()
    order = np.argsort(void, kind="mergesort")
    sorted_void = void[order]
    if sorted_void.size == 0:
        return
    change = np.flatnonzero(sorted_void[1:] != sorted_void[:-1])
    boundaries = np.concatenate((np.asarray([0], dtype=np.int64), change + 1, np.asarray([sorted_void.size], dtype=np.int64)))
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < 2:
            continue
        group = [int(index) for index in order[start:end]]
        keeper = min(group, key=lambda index: ids[index])
        for index in group:
            if index == keeper:
                continue
            status[index] = EXACT
            match_index[index] = keeper


def candidate_pairs(status: np.ndarray, initial: np.ndarray, bands: np.ndarray) -> set[tuple[int, int]]:
    """LSH pairs that cross splits, or sit inside validation or test. Train-train pairs are ignored."""
    train_idx = np.flatnonzero((status == KEPT) & (initial == TRAIN)).astype(np.int32)
    held_idx = np.flatnonzero((status == KEPT) & ((initial == VAL) | (initial == TEST)))
    train_keys: list[np.ndarray] = []
    train_docs: list[np.ndarray] = []
    for band in range(N_BANDS):
        keys = bands[band, train_idx]
        order = np.argsort(keys, kind="mergesort")
        train_keys.append(keys[order])
        train_docs.append(train_idx[order])

    pairs: set[tuple[int, int]] = set()

    def add(left: int, right: int) -> None:
        if left == right:
            return
        if left > right:
            left, right = right, left
        pairs.add((left, right))
        if len(pairs) > 1_000_000:
            raise RuntimeError(f"LSH produced more than {len(pairs)} candidate pairs.")

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for doc_index in held_idx:
        doc_index = int(doc_index)
        for band in range(N_BANDS):
            key = int(bands[band, doc_index])
            keys = train_keys[band]
            left = int(np.searchsorted(keys, np.uint64(key), side="left"))
            right = int(np.searchsorted(keys, np.uint64(key), side="right"))
            docs = train_docs[band]
            for position in range(left, right):
                add(doc_index, int(docs[position]))
            buckets[(band, key)].append(doc_index)

    for members in buckets.values():
        if len(members) < 2:
            continue
        for left_pos in range(len(members)):
            for right_pos in range(left_pos + 1, len(members)):
                add(members[left_pos], members[right_pos])
    return pairs


def apply_near(
    status: np.ndarray,
    initial: np.ndarray,
    ids: list[str],
    edges: list[tuple[int, int, float]],
    match_index: np.ndarray,
    match_jaccard: np.ndarray,
) -> None:
    """Drop held-out documents that match an anchor at or above the Jaccard threshold.

    Anchors are kept train documents, then validation documents already kept
    (in id order), then those kept validation documents when test is decided.
    A test document that matches a kept validation document is dropped too,
    so the two held-out splits do not share a near-copy. Train-train pairs stay.
    """
    neighbors: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for left, right, score in edges:
        neighbors[left].append((right, score))
        neighbors[right].append((left, score))

    anchor = (status == KEPT) & (initial == TRAIN)

    def consider(doc_index: int) -> None:
        best_index = -1
        best_score = -1.0
        best_id = None
        for other, score in neighbors.get(doc_index, ()):
            if not anchor[other]:
                continue
            other_id = ids[other]
            if score > best_score or (score == best_score and (best_id is None or other_id < best_id)):
                best_index = other
                best_score = score
                best_id = other_id
        if best_index >= 0:
            status[doc_index] = NEAR
            match_index[doc_index] = best_index
            match_jaccard[doc_index] = best_score
            return
        anchor[doc_index] = True

    val_docs = [int(index) for index in np.flatnonzero((status == KEPT) & (initial == VAL))]
    val_docs.sort(key=lambda index: ids[index])
    for doc_index in val_docs:
        consider(doc_index)

    test_docs = [int(index) for index in np.flatnonzero((status == KEPT) & (initial == TEST))]
    test_docs.sort(key=lambda index: ids[index])
    for doc_index in test_docs:
        consider(doc_index)


def assert_no_near_leak(status: np.ndarray, initial: np.ndarray, edges: list[tuple[int, int, float]]) -> None:
    for left, right, score in edges:
        if status[left] != KEPT or status[right] != KEPT:
            continue
        if initial[left] == TRAIN and initial[right] == TRAIN:
            continue
        raise RuntimeError(f"Near-duplicate still crosses splits: {left} {right} jaccard={score}")


def assert_unique_kept_sha(sha: np.ndarray, status: np.ndarray) -> None:
    kept = np.flatnonzero(status == KEPT)
    if kept.size < 2:
        return
    void = np.ascontiguousarray(sha[kept]).view(np.dtype((np.void, sha.shape[1]))).ravel()
    order = np.argsort(void, kind="mergesort")
    sorted_void = void[order]
    if np.any(sorted_void[1:] == sorted_void[:-1]):
        raise RuntimeError("Two kept documents still share an NFC text hash.")


def load_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(CHOSEN_DIR / "tokenizer.json"))
    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise SystemExit(f"Chosen vocab is {tokenizer.get_vocab_size()}, expected {VOCAB_SIZE}.")
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if eos_id != 0 or tokenizer.id_to_token(0) != EOS_TOKEN:
        raise SystemExit(f"EOS token {EOS_TOKEN!r} is id {eos_id}, expected 0.")
    probe = tokenizer.encode("Привет").ids
    if probe != [9320, 291]:
        raise SystemExit(f"Tokenizer canary changed: Привет -> {probe}.")
    if tokenizer.decode(probe) != "Привет":
        raise SystemExit("Tokenizer canary did not decode back to Привет.")
    batched = tokenizer.encode_batch(["Привет"])
    if batched[0].ids != probe:
        raise SystemExit("encode_batch does not match encode.")
    return tokenizer


def verify_tokenizer_files() -> str:
    chosen = CHOSEN_DIR / "tokenizer.json"
    reference = BPE_DIR / "tokenizer.json"
    if sha256_file(chosen) != sha256_file(reference):
        raise SystemExit("chosen/tokenizer.json does not match bpe-16000/tokenizer.json.")
    chosen_config = CHOSEN_DIR / "tokenizer_config.json"
    reference_config = BPE_DIR / "tokenizer_config.json"
    if sha256_file(chosen_config) != sha256_file(reference_config):
        raise SystemExit("chosen/tokenizer_config.json does not match bpe-16000.")
    return sha256_file(chosen)


def bind_corpus(corpus: Path) -> bool:
    """Point the module paths at a corpus directory.

    Returns True when this is the frozen FineWeb corpus, whose part hashes
    still have to match choice.json. Any other directory is checked against
    its own manifest and does not read the FineWeb fingerprint.
    """
    global CORPUS_DIR, PARTS_DIR, CORPUS_MANIFEST
    CORPUS_DIR = corpus if corpus.is_absolute() else (ROOT / corpus)
    CORPUS_DIR = CORPUS_DIR.resolve()
    PARTS_DIR = CORPUS_DIR / "parts"
    CORPUS_MANIFEST = CORPUS_DIR / "manifest.json"
    return CORPUS_DIR == FINEWEB_CORPUS.resolve()


def verify_corpus(use_frozen_choice: bool) -> dict:
    if not CORPUS_MANIFEST.exists():
        raise SystemExit(f"Missing corpus manifest: {rel(CORPUS_MANIFEST)}")
    choice_path = CHOSEN_DIR / "choice.json"
    choice = json.loads(choice_path.read_text(encoding="utf-8"))
    corpus_manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    if choice.get("name") != "bpe-16000" or int(choice.get("vocab_size", 0)) != VOCAB_SIZE:
        raise SystemExit("choice.json is not the frozen bpe-16000 tokenizer.")
    if corpus_manifest.get("status") != "complete":
        raise SystemExit("Corpus manifest is not complete.")
    if use_frozen_choice:
        fingerprint = choice["corpus"]["fingerprint"]
    else:
        fingerprint = corpus_manifest.get("parts")
        if not isinstance(fingerprint, list) or not fingerprint:
            raise SystemExit("Corpus manifest has no parts to fingerprint.")
    expected_names = [item["file"] for item in fingerprint]
    actual_names = sorted(path.name for path in PARTS_DIR.glob("part-*.parquet"))
    if actual_names != expected_names:
        raise SystemExit("Parquet part names do not match the frozen fingerprint.")
    for item in fingerprint:
        path = PARTS_DIR / item["file"]
        if path.stat().st_size != int(item["bytes"]):
            raise SystemExit(f"{item['file']} size does not match the fingerprint.")
        if sha256_file(path) != item["sha256"]:
            raise SystemExit(f"{item['file']} sha256 does not match the fingerprint.")
        rows = int(pq.ParquetFile(path).metadata.num_rows)
        if rows != int(item["rows"]):
            raise SystemExit(f"{item['file']} row count does not match the fingerprint.")
    n_docs = sum(int(item["rows"]) for item in fingerprint)
    if use_frozen_choice:
        if n_docs != int(corpus_manifest["totals"]["kept_docs"]) or n_docs != N_DOCS_EXPECTED:
            raise SystemExit(f"Frozen corpus has {n_docs} documents, expected {N_DOCS_EXPECTED}.")
    elif n_docs != int(corpus_manifest["totals"]["kept_docs"]):
        raise SystemExit(
            f"Corpus has {n_docs} documents, manifest totals say {corpus_manifest['totals']['kept_docs']}."
        )
    return {
        "choice": choice,
        "corpus_manifest": corpus_manifest,
        "fingerprint": fingerprint,
        "fingerprint_sha256": canonical_fingerprint(fingerprint),
        "part_names": expected_names,
        "part_rows": [int(item["rows"]) for item in fingerprint],
        "n_docs": n_docs,
        "n_chars": int(corpus_manifest["totals"]["chars"]),
        "revision": corpus_manifest["source"]["revision"],
    }


def self_check(space_bitmap: np.ndarray, coefficients_a: np.ndarray, coefficients_b: np.ndarray) -> None:
    probes = [
        "Привет мир это пять слов",
        "a  b\n\tc",
        "ёЁ Й\u00a0слово и ещё одно",
        "Straße И Ещё пять слов тут",
        "тест 😀 слово ещё и конец",
        "  ",
        "H2O, C6H6 и вода",
    ]
    for text in probes:
        native = word_hashes(folded_bytes(text), space_bitmap)
        reference = python_word_hashes(text)
        if not np.array_equal(native, reference):
            raise RuntimeError(f"Word hash mismatch for {text!r}")
        if not np.array_equal(shingle_hashes(native), python_shingles(reference)):
            raise RuntimeError(f"Shingle mismatch for {text!r}")

    left = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.uint64)
    right = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 11], dtype=np.uint64)
    score, ok = jaccard_pair(left, right)
    if (not ok) or abs(score - 9 / 11) > 1e-12:
        raise RuntimeError("Jaccard threshold check failed.")
    almost, almost_ok = jaccard_pair(
        np.arange(1, 11, dtype=np.uint64),
        np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 11, 12], dtype=np.uint64),
    )
    if almost_ok or abs(almost - 8 / 12) > 1e-12:
        raise RuntimeError("Jaccard below 0.8 was accepted.")

    ids = ["b", "c", "a"]
    sha = np.full((3, 32), 7, dtype=np.uint8)
    status = np.zeros(3, dtype=np.uint8)
    match_index = np.full(3, -1, dtype=np.int32)
    apply_exact(sha, ids, status, match_index)
    if status.tolist() != [EXACT, EXACT, KEPT] or match_index[:2].tolist() != [2, 2]:
        raise RuntimeError("Exact-duplicate keeper is not the smallest id.")

    def line(prefix: str, n_words: int = 40, swap_at: int | None = None, swap_to: str = "zzz") -> str:
        items = [f"{prefix}{index:02d}" for index in range(n_words)]
        if swap_at is not None:
            items[swap_at] = swap_to
        return " ".join(items)

    texts = [
        line("a"),
        line("a", swap_at=39, swap_to="zzz"),
        line("a"),
        line("a", swap_at=38, swap_to="yyy"),
        line("b"),
        line("b", swap_at=39, swap_to="qqq"),
        line("b", swap_at=39, swap_to="rrr"),
        line("c"),
    ]
    toy_ids = [f"id-{index}" for index in range(len(texts))]
    toy_initial = np.asarray([TRAIN, TRAIN, VAL, VAL, VAL, VAL, TEST, TEST], dtype=np.uint8)
    toy_status = np.zeros(len(texts), dtype=np.uint8)
    toy_match = np.full(len(texts), -1, dtype=np.int32)
    toy_jaccard = np.zeros(len(texts), dtype=np.float64)
    toy_sha = np.zeros((len(texts), 32), dtype=np.uint8)
    signatures = np.empty((len(texts), N_PERM), dtype=np.uint64)
    shingles = {}
    for index, text in enumerate(texts):
        toy_sha[index] = np.frombuffer(hashlib.sha256(nfc(text).encode("utf-8")).digest(), dtype=np.uint8)
        grams = shingles_of_text(text, space_bitmap)
        shingles[index] = grams
        signatures[index] = minhash_signature(grams, coefficients_a, coefficients_b)
    apply_exact(toy_sha, toy_ids, toy_status, toy_match)
    bands = band_hashes(signatures)
    pairs = candidate_pairs(toy_status, toy_initial, bands)
    edges = []
    for left, right in pairs:
        pair_score, pair_ok = jaccard_pair(shingles[left], shingles[right])
        if pair_ok:
            edges.append((left, right, pair_score))
    apply_near(toy_status, toy_initial, toy_ids, edges, toy_match, toy_jaccard)
    assert_no_near_leak(toy_status, toy_initial, edges)
    expected = [KEPT, KEPT, EXACT, NEAR, KEPT, NEAR, NEAR, KEPT]
    if toy_status.tolist() != expected:
        raise RuntimeError(f"Synthetic split dedup returned {toy_status.tolist()}, expected {expected}.")


def scan_documents(freeze: dict, space_bitmap: np.ndarray, coefficients_a: np.ndarray, coefficients_b: np.ndarray):
    n_docs = int(freeze["n_docs"])
    ids: list[str | None] = [None] * n_docs
    sha = np.empty((n_docs, 32), dtype=np.uint8)
    signatures = np.empty((n_docs, N_PERM), dtype=np.uint64)
    n_chars = np.empty(n_docs, dtype=np.int32)
    # uint16: a corpus of short documents can exceed 255 parts. The index is
    # only used in memory; assignments.parquet stores the part file name.
    part_of = np.empty(n_docs, dtype=np.uint16)
    row_of = np.empty(n_docs, dtype=np.int32)
    index = 0
    started = time.perf_counter()
    for part_index, (part_name, part_rows) in enumerate(zip(freeze["part_names"], freeze["part_rows"])):
        if part_index > np.iinfo(np.uint16).max:
            raise RuntimeError(f"Part index {part_index} does not fit in uint16.")
        path = PARTS_DIR / part_name
        parquet = pq.ParquetFile(path)
        row = 0
        part_started = time.perf_counter()
        for batch in parquet.iter_batches(columns=["id", "text"], batch_size=256):
            batch_ids = batch.column("id").to_pylist()
            batch_text = batch.column("text").to_pylist()
            for doc_id, text in zip(batch_ids, batch_text):
                if not isinstance(doc_id, str) or not doc_id:
                    raise RuntimeError(f"Empty id in {part_name} row {row}.")
                if not isinstance(text, str):
                    raise RuntimeError(f"Text is not a string in {part_name} row {row}.")
                normalized = nfc(text)
                sha[index] = np.frombuffer(hashlib.sha256(normalized.encode("utf-8")).digest(), dtype=np.uint8)
                raw = np.frombuffer(normalized.casefold().encode("utf-8"), dtype=np.uint8).copy()
                signatures[index] = minhash_signature(
                    shingle_hashes(word_hashes(raw, space_bitmap)),
                    coefficients_a,
                    coefficients_b,
                )
                char_len = len(text)
                if char_len > np.iinfo(np.int32).max:
                    raise RuntimeError(f"Document {doc_id} is too long to store.")
                n_chars[index] = char_len
                part_of[index] = part_index
                row_of[index] = row
                ids[index] = doc_id
                index += 1
                row += 1
            del batch, batch_ids, batch_text
        if row != part_rows:
            raise RuntimeError(f"{part_name} produced {row} rows, fingerprint says {part_rows}.")
        print(
            f"scan {part_name} rows={row} part={time.perf_counter() - part_started:.1f}s "
            f"total={time.perf_counter() - started:.0f}s rss={rss_mb():.0f}MB",
            flush=True,
        )
        gc.collect()
    if index != n_docs:
        raise RuntimeError(f"Scanned {index} documents, expected {n_docs}.")
    if int(n_chars.sum(dtype=np.int64)) != int(freeze["n_chars"]):
        raise RuntimeError("Character total does not match the corpus manifest.")
    id_order = sorted(range(n_docs), key=lambda doc_index: ids[doc_index])
    for left, right in zip(id_order, id_order[1:]):
        if ids[left] == ids[right]:
            raise RuntimeError(f"Duplicate document id {ids[left]}.")
    print(f"scan done {n_docs} docs in {time.perf_counter() - started:.0f}s", flush=True)
    return ids, sha, signatures, n_chars, part_of, row_of


def load_needed_shingles(
    freeze: dict,
    ids: list[str],
    needed: set[int],
    space_bitmap: np.ndarray,
) -> dict[int, np.ndarray]:
    if not needed:
        return {}
    found: dict[int, np.ndarray] = {}
    index = 0
    max_needed = max(needed)
    for part_name in freeze["part_names"]:
        if index > max_needed and len(found) == len(needed):
            break
        parquet = pq.ParquetFile(PARTS_DIR / part_name)
        for batch in parquet.iter_batches(columns=["id", "text"], batch_size=256):
            batch_ids = batch.column("id").to_pylist()
            batch_text = batch.column("text").to_pylist()
            for doc_id, text in zip(batch_ids, batch_text):
                if index in needed:
                    if doc_id != ids[index]:
                        raise RuntimeError("Document order changed between passes.")
                    found[index] = shingles_of_text(text, space_bitmap)
                index += 1
            del batch, batch_ids, batch_text
        gc.collect()
    if len(found) != len(needed):
        raise RuntimeError(f"Reloaded {len(found)} shingle documents, needed {len(needed)}.")
    return found


def confirm_edges(pairs: set[tuple[int, int]], shingles: dict[int, np.ndarray]) -> list[tuple[int, int, float]]:
    edges = []
    for left, right in pairs:
        score, ok = jaccard_pair(shingles[left], shingles[right])
        if ok:
            edges.append((left, right, score))
    return edges


def write_tokens(
    out_dir: Path,
    freeze: dict,
    ids: list[str],
    status: np.ndarray,
    initial: np.ndarray,
    tokenizer: Tokenizer,
    sample: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, str]]:
    n_docs = int(freeze["n_docs"])
    in_sample = np.zeros(n_docs, dtype=np.bool_)
    in_sample[sample] = True
    n_tokens = np.full(n_docs, -1, dtype=np.int64)
    token_start = np.full(n_docs, -1, dtype=np.int64)
    tokens_dir = out_dir / "tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)
    bin_paths = {name: tokens_dir / f"{name}.bin" for name in SPLIT_NAME}
    list_paths = {name: out_dir / f"{name}.txt" for name in SPLIT_NAME}
    bin_handles = {name: bin_paths[name].open("wb") for name in SPLIT_NAME}
    list_handles = {name: list_paths[name].open("w", encoding="utf-8") for name in SPLIT_NAME}
    cursors = {name: 0 for name in SPLIT_NAME}
    prefixes: dict[str, tuple[int, str]] = {}
    nfc_changed = 0
    index = 0
    started = time.perf_counter()
    try:
        for part_number, part_name in enumerate(freeze["part_names"]):
            part_started = time.perf_counter()
            parquet = pq.ParquetFile(PARTS_DIR / part_name)
            for batch in parquet.iter_batches(columns=["id", "text"], batch_size=128):
                batch_ids = batch.column("id").to_pylist()
                batch_text = batch.column("text").to_pylist()
                encodings = tokenizer.encode_batch(batch_text)
                for doc_id, text, encoding in zip(batch_ids, batch_text, encodings):
                    if doc_id != ids[index]:
                        raise RuntimeError("Document order changed between passes.")
                    token_ids = encoding.ids
                    if len(token_ids) == 0:
                        raise RuntimeError(f"Encoder returned no tokens for {doc_id}.")
                    array = np.asarray(token_ids, dtype=np.int64)
                    if int(array.min()) < 0 or int(array.max()) >= VOCAB_SIZE:
                        raise RuntimeError(f"Token id out of range in {doc_id}.")
                    if in_sample[index] or status[index] == KEPT and int(initial[index]) not in prefixes and False:
                        pass
                    check_roundtrip = bool(in_sample[index])
                    split_id = int(initial[index])
                    split_name = SPLIT_NAME[split_id]
                    first_of_split = status[index] == KEPT and split_name not in prefixes
                    if check_roundtrip or first_of_split:
                        normalized = nfc(text)
                        decoded = tokenizer.decode(token_ids)
                        if decoded != normalized:
                            raise RuntimeError(f"Roundtrip failed for {doc_id}.")
                        if check_roundtrip and normalized != text:
                            nfc_changed += 1
                    if status[index] != KEPT:
                        index += 1
                        continue
                    payload = np.empty(array.size + 1, dtype="<u2")
                    payload[:-1] = array
                    payload[-1] = np.uint16(0)
                    raw = payload.tobytes()
                    if split_name not in prefixes:
                        prefixes[split_name] = (payload.size, sha256_bytes(raw))
                    bin_handles[split_name].write(raw)
                    list_handles[split_name].write(doc_id + "\n")
                    token_start[index] = cursors[split_name]
                    n_tokens[index] = int(payload.size)
                    cursors[split_name] += int(payload.size)
                    index += 1
                del batch, batch_ids, batch_text, encodings
            print(
                f"tokenize {part_name} part={time.perf_counter() - part_started:.1f}s "
                f"total={time.perf_counter() - started:.0f}s rss={rss_mb():.0f}MB",
                flush=True,
            )
            gc.collect()
    finally:
        for handle in bin_handles.values():
            handle.close()
        for handle in list_handles.values():
            handle.close()
    if index != n_docs:
        raise RuntimeError(f"Tokenize pass saw {index} documents, expected {n_docs}.")
    if np.any((status == KEPT) & (n_tokens < 1)):
        raise RuntimeError("A kept document was not written.")
    for name, path in bin_paths.items():
        expected_tokens = int(cursors[name])
        if path.stat().st_size != expected_tokens * 2:
            raise RuntimeError(f"{name} token file has {path.stat().st_size} bytes, expected {expected_tokens * 2}.")
        if name in prefixes:
            count, digest = prefixes[name]
            with path.open("rb") as handle:
                raw = handle.read(count * 2)
            if sha256_bytes(raw) != digest:
                raise RuntimeError(f"Prefix of {name} token file does not match the first document.")
    print(f"tokenize done in {time.perf_counter() - started:.0f}s nfc_changed={nfc_changed}", flush=True)
    return n_tokens, token_start, nfc_changed, {name: sha256_file(path) for name in list(bin_paths) + list(list_paths)}


def file_record(path: Path, digest: str, **extra) -> dict:
    record = {"path": rel(path), "sha256": digest, "bytes": path.stat().st_size}
    record.update(extra)
    return record


def write_outputs(
    out_dir: Path,
    freeze: dict,
    tokenizer_sha256: str,
    ids: list[str],
    sha: np.ndarray,
    n_chars: np.ndarray,
    part_of: np.ndarray,
    row_of: np.ndarray,
    initial: np.ndarray,
    status: np.ndarray,
    match_index: np.ndarray,
    match_jaccard: np.ndarray,
    n_tokens: np.ndarray,
    token_start: np.ndarray,
    nfc_changed: int,
    file_hashes: dict[str, str],
    coefficients_a: np.ndarray,
    coefficients_b: np.ndarray,
    started_at: str,
    lsh_pairs: int,
    confirmed_edges: int,
) -> dict:
    n_docs = len(ids)
    discarded = int(np.count_nonzero(status != KEPT))
    exact = int(np.count_nonzero(status == EXACT))
    near = int(np.count_nonzero(status == NEAR))
    splits: dict[str, dict] = {}
    for split_id, name in enumerate(SPLIT_NAME):
        initial_mask = initial == split_id
        kept_mask = initial_mask & (status == KEPT)
        tokens = n_tokens[kept_mask]
        chars = n_chars[kept_mask]
        content_tokens = int(tokens.sum(dtype=np.int64)) - int(kept_mask.sum())
        splits[name] = {
            "initial_docs": int(initial_mask.sum()),
            "docs": int(kept_mask.sum()),
            "discarded_docs": int(initial_mask.sum() - kept_mask.sum()),
            "tokens": int(tokens.sum(dtype=np.int64)),
            "content_tokens": content_tokens,
            "chars": int(chars.sum(dtype=np.int64)),
            "chars_per_content_token": (float(chars.sum(dtype=np.int64)) / content_tokens) if content_tokens else 0.0,
            "length": length_stats(tokens),
        }

    sha_hex = [row.tobytes().hex() for row in sha]
    match_ids = [None if match_index[index] < 0 else ids[int(match_index[index])] for index in range(n_docs)]
    jaccard_values = [float(match_jaccard[index]) if status[index] == NEAR else None for index in range(n_docs)]
    initial_names = [SPLIT_NAME[int(initial[index])] for index in range(n_docs)]
    final_names = [initial_names[index] if status[index] == KEPT else "" for index in range(n_docs)]
    status_names = [STATUS_NAME[int(status[index])] for index in range(n_docs)]
    part_names = [freeze["part_names"][int(part_of[index])] for index in range(n_docs)]
    kept_null = status != KEPT

    table = pa.table(
        {
            "id": pa.array(ids, type=pa.string()),
            "part": pa.array(part_names, type=pa.string()),
            "row": pa.array(row_of, type=pa.int32()),
            "initial_split": pa.array(initial_names, type=pa.string()),
            "split": pa.array(final_names, type=pa.string()),
            "status": pa.array(status_names, type=pa.string()),
            "text_sha256": pa.array(sha_hex, type=pa.string()),
            "token_start": pa.array(token_start, mask=kept_null),
            "n_tokens": pa.array(n_tokens, mask=kept_null),
            "match_id": pa.array(match_ids, type=pa.string()),
            "jaccard": pa.array(jaccard_values, type=pa.float64()),
        }
    )
    assignments_path = out_dir / "assignments.parquet"
    temporary_assignments = assignments_path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary_assignments, compression="zstd")
    os.replace(temporary_assignments, assignments_path)

    duplicates_path = out_dir / "duplicates.jsonl"
    with duplicates_path.open("w", encoding="utf-8") as handle:
        for index in range(n_docs):
            if status[index] == KEPT:
                continue
            keeper = int(match_index[index])
            record = {
                "kind": "exact" if status[index] == EXACT else "near",
                "id": ids[index],
                "initial_split": initial_names[index],
                "match_id": ids[keeper],
                "match_initial_split": SPLIT_NAME[int(initial[keeper])],
                "action": "discard",
            }
            if status[index] == EXACT:
                record["sha256"] = sha_hex[index]
            else:
                record["jaccard"] = float(match_jaccard[index])
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    content_tokens = sum(item["content_tokens"] for item in splits.values())
    tokens_with_eos = sum(item["tokens"] for item in splits.values())
    kept_docs = n_docs - discarded
    manifest = {
        "name": f"{freeze['corpus_manifest']['name']}-dataset",
        "status": "complete",
        "description": (
            "Зафиксированный учебный датасет: BPE 16 000, разбиение 98/1/1 сидом 22, "
            "поток uint16 с EOS после каждого документа. Ноль в файле — граница документа, не паддинг."
        ),
        "created_at": started_at,
        "finished_at": now_iso(),
        "script": rel(Path(__file__)),
        "command": " ".join(sys.argv),
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
        "tokenizers": __import__("tokenizers").__version__,
        "freeze": {
            "tokenizer": rel(CHOSEN_DIR),
            "tokenizer_sha256": tokenizer_sha256,
            "matches_bpe_16000": True,
            "vocab_size": VOCAB_SIZE,
            "eos_token": EOS_TOKEN,
            "eos_id": 0,
            "normalizer": "NFC",
            "corpus_fingerprint_sha256": freeze["fingerprint_sha256"],
            "corpus_manifest": rel(CORPUS_MANIFEST),
            "choice": rel(CHOSEN_DIR / "choice.json"),
            "revision": freeze["revision"],
            "filter": freeze["corpus_manifest"]["filter"]["name"],
            "corpus_seed": int(freeze["corpus_manifest"]["sampling"]["seed"]),
            "docs": n_docs,
            "chars": int(freeze["n_chars"]),
        },
        "split": {
            "seed": SEED,
            "generator": "numpy.random.default_rng",
            "n_docs": n_docs,
            "validation_count": n_docs // 100,
            "test_count": n_docs // 100,
            "rule": "permutation[:n//100] validation, next n//100 test, the rest train",
            "order_within_split": "corpus order: part file name, then row",
            "refill": False,
        },
        "dedup": {
            "exact": "sha256 of NFC text; keep the lexicographically smallest id; drop the other copies from every split",
            "near": {
                "method": "minhash",
                "shingle": "5 words, regex \\S+ on NFC casefold",
                "permutations": N_PERM,
                "bands": N_BANDS,
                "rows_per_band": BAND_ROWS,
                "jaccard_min": JACCARD_NUM / JACCARD_DEN,
                "seed": SEED,
                "coefficients_a": [format(int(value), "016x") for value in coefficients_a],
                "coefficients_b": [format(int(value), "016x") for value in coefficients_b],
                "policy": (
                    "Drop a validation or test document if it matches a kept train document, "
                    "an already kept document of the same held-out split, or, for test, a kept "
                    "validation document. Near-copies inside train are kept. Held-out splits are not refilled."
                ),
                "lsh_pairs": lsh_pairs,
                "confirmed_edges": confirmed_edges,
            },
        },
        "tokens": {
            "dtype": "uint16",
            "endian": "little",
            "eos_after_each_document": True,
            "padding": False,
            "packed_context": False,
            "content_tokens": content_tokens,
            "tokens_with_eos": tokens_with_eos,
        },
        "totals": {
            "docs": n_docs,
            "kept_docs": kept_docs,
            "discarded_docs": discarded,
            "discarded_exact": exact,
            "discarded_near": near,
            "discarded_fraction": discarded / n_docs,
        },
        "splits": splits,
        "roundtrip": {
            "docs": ROUNDTRIP_DOCS,
            "seed": SEED,
            "generator": "numpy.random.default_rng",
            "nfc_changed": nfc_changed,
            "failures": 0,
            "compare": "decode(encode(text)) == NFC(text)",
        },
        "files": {},
    }
    files = manifest["files"]
    files["assignments"] = file_record(assignments_path, sha256_file(assignments_path), rows=n_docs)
    files["duplicates"] = file_record(duplicates_path, sha256_file(duplicates_path))
    for name in SPLIT_NAME:
        list_path = out_dir / f"{name}.txt"
        bin_path = out_dir / "tokens" / f"{name}.bin"
        files[f"{name}_ids"] = file_record(list_path, file_hashes[name] if False else sha256_file(list_path), lines=splits[name]["docs"])
        files[f"{name}_tokens"] = file_record(
            bin_path,
            sha256_file(bin_path),
            tokens=splits[name]["tokens"],
        )
    # file_hashes from the tokenize pass are unused once the files are closed and rehashed.
    # Rehashing catches a partial write. The prefix check already compared the first document.
    del file_hashes

    manifest_path = out_dir / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return manifest


def dataset_is_current(out_dir: Path, freeze: dict, tokenizer_sha256: str) -> dict | None:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        if out_dir.exists() and any(out_dir.iterdir()):
            raise SystemExit(f"{rel(out_dir)} is incomplete. Pass --refresh to rebuild.")
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise SystemExit(f"{rel(out_dir)} is incomplete. Pass --refresh to rebuild.")
    stored = manifest.get("freeze", {})
    if stored.get("tokenizer_sha256") != tokenizer_sha256 or stored.get("corpus_fingerprint_sha256") != freeze["fingerprint_sha256"]:
        raise SystemExit("Frozen tokenizer or corpus changed after the dataset was built. Pass --refresh to rebuild.")
    return manifest


def prepare(out_dir: Path, refresh: bool, use_frozen_choice: bool) -> None:
    started_at = now_iso()
    tokenizer_sha256 = verify_tokenizer_files()
    tokenizer = load_tokenizer()
    print("tokenizer freeze ok", flush=True)
    space_bitmap = build_space_bitmap()
    coefficients_a, coefficients_b = minhash_coefficients(SEED)
    self_check(space_bitmap, coefficients_a, coefficients_b)
    print("self-check ok", flush=True)
    freeze = verify_corpus(use_frozen_choice)
    print("corpus fingerprint ok", flush=True)
    if refresh and out_dir.exists():
        shutil.rmtree(out_dir)
    elif not refresh:
        existing = dataset_is_current(out_dir, freeze, tokenizer_sha256)
        if existing is not None:
            print(json.dumps({"manifest": rel(out_dir / "manifest.json"), **existing["totals"]}, ensure_ascii=False, indent=2))
            return
    out_dir.mkdir(parents=True, exist_ok=True)

    ids, sha, signatures, n_chars, part_of, row_of = scan_documents(
        freeze, space_bitmap, coefficients_a, coefficients_b
    )
    initial = assign_splits(int(freeze["n_docs"]), SEED)
    n_val = int(np.count_nonzero(initial == VAL))
    n_test = int(np.count_nonzero(initial == TEST))
    if n_val != freeze["n_docs"] // 100 or n_test != freeze["n_docs"] // 100:
        raise RuntimeError("Split sizes are not n//100.")
    print(f"split train={freeze['n_docs'] - n_val - n_test} validation={n_val} test={n_test}", flush=True)

    status = np.zeros(freeze["n_docs"], dtype=np.uint8)
    match_index = np.full(freeze["n_docs"], -1, dtype=np.int32)
    match_jaccard = np.zeros(freeze["n_docs"], dtype=np.float64)
    apply_exact(sha, ids, status, match_index)
    assert_unique_kept_sha(sha, status)
    exact = int(np.count_nonzero(status == EXACT))
    print(f"exact duplicates discarded={exact}", flush=True)

    bands = band_hashes(signatures)
    del signatures
    gc.collect()
    pairs = candidate_pairs(status, initial, bands)
    n_lsh_pairs = len(pairs)
    del bands
    gc.collect()
    print(f"lsh pairs={n_lsh_pairs} rss={rss_mb():.0f}MB", flush=True)
    needed = {doc for pair in pairs for doc in pair}
    shingles = load_needed_shingles(freeze, ids, needed, space_bitmap)
    edges = confirm_edges(pairs, shingles)
    del shingles, pairs
    gc.collect()
    print(f"near edges={len(edges)}", flush=True)
    apply_near(status, initial, ids, edges, match_index, match_jaccard)
    assert_no_near_leak(status, initial, edges)
    assert_unique_kept_sha(sha, status)
    print(
        f"discarded exact={exact} near={int(np.count_nonzero(status == NEAR))} "
        f"kept={int(np.count_nonzero(status == KEPT))}",
        flush=True,
    )

    sample = np.random.default_rng(SEED).choice(freeze["n_docs"], size=ROUNDTRIP_DOCS, replace=False)
    n_tokens, token_start, nfc_changed, file_hashes = write_tokens(
        out_dir, freeze, ids, status, initial, tokenizer, sample
    )
    manifest = write_outputs(
        out_dir,
        freeze,
        tokenizer_sha256,
        ids,
        sha,
        n_chars,
        part_of,
        row_of,
        initial,
        status,
        match_index,
        match_jaccard,
        n_tokens,
        token_start,
        nfc_changed,
        file_hashes,
        coefficients_a,
        coefficients_b,
        started_at,
        lsh_pairs=n_lsh_pairs,
        confirmed_edges=len(edges),
    )
    print(json.dumps({"manifest": rel(out_dir / "manifest.json"), **manifest["totals"], "splits": {name: {"docs": item["docs"], "tokens": item["tokens"]} for name, item in manifest["splits"].items()}}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=FINEWEB_CORPUS, help="Corpus directory with manifest.json and parts/.")
    parser.add_argument("--out", type=Path, default=None, help="Dataset directory. Default: <corpus>/dataset.")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    use_frozen_choice = bind_corpus(args.corpus)
    if args.out is None:
        out_dir = CORPUS_DIR / "dataset"
    else:
        out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    prepare(out_dir, args.refresh, use_frozen_choice)


if __name__ == "__main__":
    main()
