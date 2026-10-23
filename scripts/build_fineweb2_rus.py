#!/usr/bin/env python3
"""Build a local FineWeb-2 Russian corpus for a 50-100M model.

Reads parquet row groups from HuggingFaceFW/fineweb-2 rus_Cyrl over HTTP,
keeps documents that pass the same strict_keep screen as notebooks/eda.ipynb,
and stops near 1e9 estimated tokens (characters / 2.5). That is about 20
tokens per parameter for a 50M model.

Outputs:
  data/fineweb2/rus_Cyrl/v001/parts/part-*.parquet
  data/fineweb2/rus_Cyrl/v001/manifest.json

Load with pandas.read_parquet("data/fineweb2/rus_Cyrl/v001/parts").

The text stays as strings. Tokenization is a later step. Safe to rerun:
finished row groups are skipped. --refresh deletes v001 and starts over.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "fineweb2" / "rus_Cyrl" / "v001"
REPO = "HuggingFaceFW/fineweb-2"
REMOTE_DIR = "data/rus_Cyrl/train"

TARGET_TOKENS = 1_000_000_000
CHARS_PER_TOKEN = 2.5
SEED = 22
GROUPS_PER_FILE = 2
PART_ROWS = 20_000
RESERVOIR_SIZE = 10_000

CYR_MIN = 0.90
WORDS_MIN = 80
NEWLINE_MAX = 0.08
DUP_LINE_MAX = 0.15
PUNCT_MIN = 0.20
LANG_MIN = 0.95

WORD_RE = re.compile(r"\S+")
TERMINAL_PUNCT = set(".!?…:;")
KEEP_COLUMNS = [
    "id",
    "text",
    "url",
    "dump",
    "date",
    "language",
    "language_script",
    "language_score",
    "minhash_cluster_size",
    "source_file",
    "row_group",
]
READ_COLUMNS = [
    "id",
    "text",
    "url",
    "dump",
    "date",
    "language",
    "language_script",
    "language_score",
    "minhash_cluster_size",
]
SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("text", pa.string()),
        ("url", pa.string()),
        ("dump", pa.string()),
        ("date", pa.string()),
        ("language", pa.string()),
        ("language_script", pa.string()),
        ("language_score", pa.float64()),
        ("minhash_cluster_size", pa.int64()),
        ("source_file", pa.string()),
        ("row_group", pa.int32()),
    ]
)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_seed(path: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}\n{path}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def remote_url(path: str, revision: str) -> str:
    return f"hf://datasets/{REPO}@{revision}/{path}"


def list_train_files(revision: str) -> list[str]:
    from huggingface_hub import HfApi

    entries = HfApi().list_repo_tree(
        REPO,
        repo_type="dataset",
        path_in_repo=REMOTE_DIR,
        revision=revision,
        recursive=False,
    )
    files = sorted(item.path for item in entries if str(item.path).endswith(".parquet"))
    if not files:
        raise RuntimeError(f"No parquet files under {REMOTE_DIR} at {revision}")
    return files


def resolve_revision(revision: str | None) -> str:
    if revision:
        return revision
    from huggingface_hub import HfApi

    info = HfApi().repo_info(REPO, repo_type="dataset")
    return info.sha


def signals(text: str) -> tuple[int, int, float, float, float, float]:
    """n_chars, n_words, cyr_ratio, newline_ratio, dup_line_char_frac, punct_line_frac."""
    words = WORD_RE.findall(text)
    n_words = len(words)
    n_chars = len(text)
    n_cyr = 0
    n_letters = 0
    for ch in text:
        if not ch.isalpha():
            continue
        n_letters += 1
        if 0x0400 <= ord(ch) <= 0x04FF:
            n_cyr += 1
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        counts = Counter(lines)
        line_chars = sum(len(line) * count for line, count in counts.items())
        dup_chars = sum(len(line) * count for line, count in counts.items() if count > 1)
        dup_line = dup_chars / line_chars if line_chars else 0.0
        punct = sum(line[-1] in TERMINAL_PUNCT for line in lines) / len(lines)
    else:
        dup_line = 0.0
        punct = 0.0
    cyr_ratio = (n_cyr / n_letters) if n_letters else 0.0
    newline_ratio = (text.count("\n") / n_chars) if n_chars else 0.0
    return n_chars, n_words, cyr_ratio, newline_ratio, dup_line, punct


def choose_row_groups(path: str, n_groups: int, k: int, seed: int) -> list[int]:
    rng = np.random.default_rng(file_seed(path, seed))
    take = min(k, n_groups)
    picked = rng.choice(n_groups, size=take, replace=False)
    return sorted(int(value) for value in np.atleast_1d(picked))


def load_file(path: str, groups_per_file: int, seed: int, revision: str) -> tuple[pd.DataFrame, list[int], int]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            pf = pq.ParquetFile(remote_url(path, revision))
            n_groups = int(pf.num_row_groups)
            groups = choose_row_groups(path, n_groups, groups_per_file, seed)
            frames = []
            for group in groups:
                frame = pf.read_row_group(group, columns=READ_COLUMNS).to_pandas()
                frame["row_group"] = np.int32(group)
                frames.append(frame)
            return pd.concat(frames, ignore_index=True), groups, n_groups
        except Exception as exc:  # network reads fail transiently
            last_error = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Failed to read {path}") from last_error


def filter_frame(frame: pd.DataFrame, source_file: str, groups: list[int]) -> tuple[pd.DataFrame, dict]:
    n_chars: list[int] = []
    n_words: list[int] = []
    cyr_ratio: list[float] = []
    newline_ratio: list[float] = []
    dup_line: list[float] = []
    punct: list[float] = []
    texts = frame["text"].tolist()
    for text in texts:
        if not isinstance(text, str) or not text:
            n_chars.append(0)
            n_words.append(0)
            cyr_ratio.append(0.0)
            newline_ratio.append(1.0)
            dup_line.append(1.0)
            punct.append(0.0)
            continue
        chars, words, cyr, newlines, dup, punct_frac = signals(text)
        n_chars.append(chars)
        n_words.append(words)
        cyr_ratio.append(cyr)
        newline_ratio.append(newlines)
        dup_line.append(dup)
        punct.append(punct_frac)

    cyr_ok = np.asarray(cyr_ratio) >= CYR_MIN
    words_ok = np.asarray(n_words) >= WORDS_MIN
    newline_ok = np.asarray(newline_ratio) <= NEWLINE_MAX
    dup_ok = np.asarray(dup_line) <= DUP_LINE_MAX
    punct_ok = np.asarray(punct) >= PUNCT_MIN
    lang = pd.to_numeric(frame["language_score"], errors="coerce").fillna(0.0).to_numpy()
    lang_ok = lang >= LANG_MIN
    keep = cyr_ok & words_ok & newline_ok & dup_ok & punct_ok & lang_ok

    kept = frame.loc[keep].copy()
    kept["source_file"] = source_file
    kept["n_chars"] = np.asarray(n_chars)[keep]
    kept["n_words"] = np.asarray(n_words)[keep]
    kept = kept.reset_index(drop=True)

    stats = {
        "read_docs": int(len(frame)),
        "kept_docs": int(keep.sum()),
        "kept_chars": int(np.asarray(n_chars)[keep].sum()) if keep.any() else 0,
        "kept_words": int(np.asarray(n_words)[keep].sum()) if keep.any() else 0,
        "fail_cyr_ratio": int((~cyr_ok).sum()),
        "fail_n_words": int((~words_ok).sum()),
        "fail_newline_ratio": int((~newline_ok).sum()),
        "fail_dup_line_char_frac": int((~dup_ok).sum()),
        "fail_punct_line_frac": int((~punct_ok).sum()),
        "fail_language_score": int((~lang_ok).sum()),
        "minhash_eq_1": int((kept["minhash_cluster_size"] == 1).sum()) if len(kept) else 0,
        "minhash_sum": int(pd.to_numeric(kept["minhash_cluster_size"], errors="coerce").fillna(0).sum()) if len(kept) else 0,
        "language_score_sum": float(pd.to_numeric(kept["language_score"], errors="coerce").fillna(0).sum()) if len(kept) else 0.0,
    }
    return kept[KEEP_COLUMNS + ["n_chars", "n_words"]], stats


def domain_of(url: str) -> str:
    if not isinstance(url, str):
        return "(none)"
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host or "(none)"


def rng_from_state(payload: dict, seed: int) -> random.Random:
    rng = random.Random(seed)
    raw = payload.get("rng_state")
    if not raw:
        return rng
    version, internal, gauss = raw
    rng.setstate((version, tuple(internal), gauss))
    return rng


def rng_to_state(rng: random.Random) -> list:
    version, internal, gauss = rng.getstate()
    return [version, list(internal), gauss]


def reservoir_add(reservoir: list[dict], item: dict, seen: int, rng: random.Random) -> None:
    if len(reservoir) < RESERVOIR_SIZE:
        reservoir.append(item)
        return
    slot = rng.randrange(seen)
    if slot < RESERVOIR_SIZE:
        reservoir[slot] = item


def load_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_part(parts_dir: Path, part_index: int, frame: pd.DataFrame) -> dict:
    parts_dir.mkdir(parents=True, exist_ok=True)
    name = f"part-{part_index:05d}.parquet"
    destination = parts_dir / name
    temporary = parts_dir / f"{name}.tmp"
    table = pa.Table.from_pandas(frame[KEEP_COLUMNS], schema=SCHEMA, preserve_index=False)
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, destination)
    return {
        "file": name,
        "rows": int(len(frame)),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values)
    keys = [10, 25, 50, 75, 90, 99]
    points = np.percentile(array, keys)
    return {f"p{key}": float(point) for key, point in zip(keys, points)}


def write_manifest(out_dir: Path, state: dict) -> Path:
    reservoir = state.get("reservoir", [])
    chars = [item["n_chars"] for item in reservoir]
    words = [item["n_words"] for item in reservoir]
    kept_docs = int(state["kept_docs"])
    read_docs = int(state["read_docs"])
    parts_dir = out_dir / "parts"
    try:
        load_path = str(parts_dir.relative_to(ROOT))
    except ValueError:
        load_path = str(parts_dir)
    manifest = {
        "name": "fineweb2-rus-v001",
        "status": "complete",
        "description": (
            "Русский учебный корпус для модели на 50–100M параметров. "
            "Объём около 1 млрд токенов: это примерно 20 токенов на параметр для модели 50M "
            "и 10 токенов на параметр для модели 100M. Токены пока оценены как символы / 2.5, "
            "до появления отдельного русского токенизатора. Текст сохранён строками."
        ),
        "created_at": state["started_at"],
        "finished_at": now_iso(),
        "script": "scripts/build_fineweb2_rus.py",
        "command": state["command"],
        "python": sys.version.split()[0],
        "pyarrow": pa.__version__,
        "source": {
            "repo": REPO,
            "subset": "rus_Cyrl",
            "split": "train",
            "revision": state["revision"],
            "license": "ODC-By-1.0",
            "full_subset_documents": 699_083_579,
            "full_subset_disk": "1.81TB",
        },
        "sizing": {
            "model_parameters": "50-100M",
            "target_tokens": state["target_tokens"],
            "chars_per_token_estimate": CHARS_PER_TOKEN,
            "estimated_tokens": int(state["kept_chars"] / CHARS_PER_TOKEN),
            "actual_may_exceed_target_by_one_file": True,
        },
        "sampling": {
            "seed": state["seed"],
            "groups_per_file": state["groups_per_file"],
            "files_read": len(state["sources"]),
            "duplicate_upsampling": False,
            "note": (
                "Train files are shuffled with the seed. Each opened file contributes "
                "groups_per_file row groups, chosen by a hash of the file path. "
                "FineWeb-2 duplicate upsampling weights are not applied."
            ),
        },
        "filter": {
            "name": "strict_keep",
            "same_as": "notebooks/eda.ipynb",
            "cyr_ratio_min": CYR_MIN,
            "n_words_min": WORDS_MIN,
            "newline_ratio_max": NEWLINE_MAX,
            "dup_line_char_frac_max": DUP_LINE_MAX,
            "punct_line_frac_min": PUNCT_MIN,
            "language_score_min": LANG_MIN,
        },
        "totals": {
            "read_docs": read_docs,
            "kept_docs": kept_docs,
            "rejected_docs": int(state["rejected_docs"]),
            "keep_rate": (kept_docs / read_docs) if read_docs else 0.0,
            "chars": int(state["kept_chars"]),
            "words": int(state["kept_words"]),
            "estimated_tokens": int(state["kept_chars"] / CHARS_PER_TOKEN),
            "mean_language_score": (state["language_score_sum"] / kept_docs) if kept_docs else 0.0,
            "minhash_cluster_size_mean": (state["minhash_sum"] / kept_docs) if kept_docs else 0.0,
            "minhash_cluster_size_eq_1": int(state["minhash_eq_1"]),
            "domains": int(state.get("domains", 0)),
        },
        "rejection_counts": state["fail_counts"],
        "length_quantiles_from_reservoir": {
            "reservoir_size": len(reservoir),
            "chars": quantiles(chars),
            "words": quantiles(words),
        },
        "load": f"pandas.read_parquet({load_path!r})",
        "columns": KEEP_COLUMNS,
        "parts": state["parts"],
        "sources": state["sources"],
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def count_domains(parts_dir: Path, parts: list[dict]) -> int:
    hosts: set[str] = set()
    for part in parts:
        table = pq.read_table(parts_dir / part["file"], columns=["url"])
        for url in table.column("url").to_pylist():
            hosts.add(domain_of(url))
    return len(hosts)


def new_state(revision: str, seed: int, target_tokens: int, groups_per_file: int, file_order: list[str], command: str) -> dict:
    rng = random.Random(seed)
    return {
        "revision": revision,
        "seed": seed,
        "target_tokens": target_tokens,
        "groups_per_file": groups_per_file,
        "file_order": file_order,
        "command": command,
        "started_at": now_iso(),
        "completed_files": [],
        "sources": [],
        "parts": [],
        "kept_docs": 0,
        "kept_chars": 0,
        "kept_words": 0,
        "read_docs": 0,
        "rejected_docs": 0,
        "fail_counts": {
            "cyr_ratio": 0,
            "n_words": 0,
            "newline_ratio": 0,
            "dup_line_char_frac": 0,
            "punct_line_frac": 0,
            "language_score": 0,
        },
        "minhash_eq_1": 0,
        "minhash_sum": 0,
        "language_score_sum": 0.0,
        "reservoir": [],
        "rng_state": rng_to_state(rng),
    }


def build(out_dir: Path, target_tokens: int, seed: int, groups_per_file: int, revision: str | None, refresh: bool) -> None:
    manifest_path = out_dir / "manifest.json"
    state_path = out_dir / "_state.json"
    if refresh and out_dir.exists():
        shutil.rmtree(out_dir)
    if manifest_path.exists() and not refresh:
        print(f"already built: {manifest_path}")
        return

    revision = resolve_revision(revision)
    out_dir.mkdir(parents=True, exist_ok=True)
    for temporary in out_dir.glob("*.tmp"):
        temporary.unlink()

    state = load_state(state_path)
    if state is None:
        files = list_train_files(revision)
        order_rng = np.random.default_rng(seed)
        permutation = order_rng.permutation(len(files))
        file_order = [files[int(index)] for index in permutation]
        state = new_state(
            revision=revision,
            seed=seed,
            target_tokens=target_tokens,
            groups_per_file=groups_per_file,
            file_order=file_order,
            command=" ".join(sys.argv),
        )
        save_state(state_path, state)
        print(f"revision {revision}  train files {len(file_order):,}", flush=True)
    else:
        if state["seed"] != seed or state["target_tokens"] != target_tokens:
            raise SystemExit(
                "v001 already has a different seed or target. Use --refresh or another --out directory."
            )
        revision = state["revision"]
        print(
            f"resume kept_docs={state['kept_docs']:,} estimated_tokens={int(state['kept_chars'] / CHARS_PER_TOKEN):,}",
            flush=True,
        )

    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    known_parts = {part["file"] for part in state["parts"]}
    for path in parts_dir.glob("part-*.parquet"):
        if path.name not in known_parts:
            path.unlink()

    completed = set(state["completed_files"])
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    rng = rng_from_state(state, seed)
    buffer: list[pd.DataFrame] = []
    buffer_rows = 0
    started = time.time()

    pending_sources: list[dict] = []

    def flush() -> None:
        nonlocal buffer, buffer_rows, pending_sources
        if buffer_rows:
            frame = pd.concat(buffer, ignore_index=True)
            part = write_part(parts_dir, len(state["parts"]), frame)
            state["parts"].append(part)
            print(f"  wrote {part['file']} rows={part['rows']:,} bytes={part['bytes']:,}", flush=True)
            buffer = []
            buffer_rows = 0
        if not pending_sources:
            return
        for source in pending_sources:
            state["sources"].append(source)
            state["completed_files"].append(source["source_file"])
            completed.add(source["source_file"])
        pending_sources = []
        state["rng_state"] = rng_to_state(rng)
        save_state(state_path, state)

    for path in state["file_order"]:
        if int(state["kept_chars"]) >= target_chars:
            break
        if path in completed:
            continue
        frame, groups, n_groups = load_file(path, groups_per_file, seed, revision)
        kept, stats = filter_frame(frame, path, groups)
        state["read_docs"] += stats["read_docs"]
        state["rejected_docs"] += stats["read_docs"] - stats["kept_docs"]
        state["kept_docs"] += stats["kept_docs"]
        state["kept_chars"] += stats["kept_chars"]
        state["kept_words"] += stats["kept_words"]
        state["minhash_eq_1"] += stats["minhash_eq_1"]
        state["minhash_sum"] += stats["minhash_sum"]
        state["language_score_sum"] += stats["language_score_sum"]
        for key in state["fail_counts"]:
            state["fail_counts"][key] += stats[f"fail_{key}"]
        seen = state["kept_docs"] - stats["kept_docs"]
        for chars, words in zip(kept["n_chars"].tolist(), kept["n_words"].tolist()):
            seen += 1
            reservoir_add(state["reservoir"], {"n_chars": int(chars), "n_words": int(words)}, seen, rng)
        if len(kept):
            buffer.append(kept[KEEP_COLUMNS])
            buffer_rows += len(kept)
        pending_sources.append(
            {
                "source_file": path,
                "row_groups": groups,
                "n_row_groups": n_groups,
                "read_docs": stats["read_docs"],
                "kept_docs": stats["kept_docs"],
            }
        )
        if buffer_rows >= PART_ROWS:
            flush()
        elapsed = time.time() - started
        rate = state["kept_chars"] / elapsed if elapsed else 0
        remaining = max(target_chars - state["kept_chars"], 0)
        eta_min = (remaining / rate / 60) if rate else 0
        print(
            f"{path} groups={groups} kept={stats['kept_docs']}/{stats['read_docs']} "
            f"tokens={int(state['kept_chars'] / CHARS_PER_TOKEN):,}/{target_tokens:,} "
            f"eta={eta_min:.0f}min",
            flush=True,
        )

    flush()
    if int(state["kept_chars"]) < target_chars:
        raise SystemExit(
            f"Ran out of files at {int(state['kept_chars'] / CHARS_PER_TOKEN):,} estimated tokens."
        )
    print("counting domains", flush=True)
    state["domains"] = count_domains(parts_dir, state["parts"])
    manifest = write_manifest(out_dir, state)
    totals = json.loads(manifest.read_text(encoding="utf-8"))["totals"]
    try:
        shown = str(manifest.relative_to(ROOT))
    except ValueError:
        shown = str(manifest)
    print(json.dumps({"manifest": shown, **totals}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--groups-per-file", type=int, default=GROUPS_PER_FILE)
    parser.add_argument("--revision", default=None, help="Hugging Face dataset revision. Default: current main sha, then frozen in the build.")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    build(
        out_dir=args.out,
        target_tokens=args.target_tokens,
        seed=args.seed,
        groups_per_file=args.groups_per_file,
        revision=args.revision,
        refresh=args.refresh,
    )


if __name__ == "__main__":
    main()
