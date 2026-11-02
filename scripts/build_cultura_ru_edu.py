#!/usr/bin/env python3
"""Build a local Cultura-Ru-Edu slice the same size as fineweb2-rus-v001.

Reads train parquet shards from deepvk/cultura_ru_edu over HTTP, keeps
documents that pass the FineWeb strict_keep text screen except language_score
(this dataset has no language score), and stops at the FineWeb v001 character
count. The Hugging Face validation split is not used.

Outputs:
  data/cultura_ru_edu/v001/parts/part-*.parquet
  data/cultura_ru_edu/v001/manifest.json

Safe to rerun: a finished manifest is left untouched. An interrupted build
resumes from the last flushed part. --refresh deletes the output and starts over.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from build_fineweb2_rus import (
    CYR_MIN,
    DUP_LINE_MAX,
    NEWLINE_MAX,
    PUNCT_MIN,
    RESERVOIR_SIZE,
    WORDS_MIN,
    now_iso,
    quantiles,
    reservoir_add,
    rng_from_state,
    rng_to_state,
    sha256_file,
    signals,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "cultura_ru_edu" / "v001"
REPO = "deepvk/cultura_ru_edu"
# Character count of data/fineweb2/rus_Cyrl/v001, so the same BPE sees a similar token budget.
TARGET_CHARS = 2_504_449_346
SEED = 22
PART_ROWS = 20_000
BATCH_ROWS = 256

SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("text", pa.string()),
        ("source_file", pa.string()),
        ("row_group", pa.int32()),
        ("row", pa.int32()),
    ]
)
FAIL_KEYS = (
    "cyr_ratio",
    "n_words",
    "newline_ratio",
    "dup_line_char_frac",
    "punct_line_frac",
)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def list_train_files(revision: str) -> list[str]:
    from huggingface_hub import HfApi

    entries = HfApi().list_repo_tree(
        REPO,
        repo_type="dataset",
        path_in_repo="data",
        revision=revision,
        recursive=False,
    )
    files = sorted(
        item.path
        for item in entries
        if str(item.path).endswith(".parquet") and Path(str(item.path)).name.startswith("train-")
    )
    if not files:
        raise RuntimeError(f"No train parquet files in {REPO} at {revision}")
    return files


def resolve_revision(revision: str | None) -> str:
    if revision:
        return revision
    from huggingface_hub import HfApi

    info = HfApi().repo_info(REPO, repo_type="dataset")
    return info.sha


def download_shard(path: str, revision: str) -> str:
    from huggingface_hub import hf_hub_download

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return hf_hub_download(
                repo_id=REPO,
                filename=path,
                repo_type="dataset",
                revision=revision,
            )
        except Exception as exc:  # network reads fail transiently
            last_error = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Failed to download {path}") from last_error


def screen(text: object) -> tuple[bool, int, int, dict[str, int]]:
    """Return keep, n_chars, n_words, and which thresholds failed."""
    if not isinstance(text, str) or not text:
        chars, words, cyr, newlines, dup, punct = 0, 0, 0.0, 1.0, 1.0, 0.0
    else:
        chars, words, cyr, newlines, dup, punct = signals(text)
    fails = {
        "cyr_ratio": int(cyr < CYR_MIN),
        "n_words": int(words < WORDS_MIN),
        "newline_ratio": int(newlines > NEWLINE_MAX),
        "dup_line_char_frac": int(dup > DUP_LINE_MAX),
        "punct_line_frac": int(punct < PUNCT_MIN),
    }
    return sum(fails.values()) == 0, chars, words, fails


def load_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def new_state(revision: str, seed: int, target_chars: int, file_order: list[str], command: str) -> dict:
    return {
        "revision": revision,
        "seed": seed,
        "target_chars": target_chars,
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
        "fail_counts": {key: 0 for key in FAIL_KEYS},
        "reservoir": [],
        "rng_state": None,
        "cursor_file": None,
        "cursor_row": 0,
        "cursor_read_docs": 0,
        "cursor_kept_docs": 0,
    }


def write_part(parts_dir: Path, part_index: int, rows: dict[str, list]) -> dict:
    parts_dir.mkdir(parents=True, exist_ok=True)
    name = f"part-{part_index:05d}.parquet"
    destination = parts_dir / name
    temporary = parts_dir / f"{name}.tmp"
    table = pa.table(
        {
            "id": pa.array(rows["id"], type=pa.string()),
            "text": pa.array(rows["text"], type=pa.string()),
            "source_file": pa.array(rows["source_file"], type=pa.string()),
            "row_group": pa.array(rows["row_group"], type=pa.int32()),
            "row": pa.array(rows["row"], type=pa.int32()),
        },
        schema=SCHEMA,
    )
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, destination)
    return {
        "file": name,
        "rows": int(table.num_rows),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def empty_buffer() -> dict[str, list]:
    return {"id": [], "text": [], "source_file": [], "row_group": [], "row": []}


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
        "name": f"cultura-ru-edu-{out_dir.name}",
        "status": "complete",
        "description": (
            "Русский учебный срез Cultura-Ru-Edu. "
            f"Остановка на {int(state['target_chars'])} символах, "
            "как у fineweb2-rus-v001, чтобы тот же BPE дал близкий объём токенов."
        ),
        "created_at": state["started_at"],
        "finished_at": now_iso(),
        "script": "scripts/build_cultura_ru_edu.py",
        "command": state["command"],
        "python": sys.version.split()[0],
        "pyarrow": pa.__version__,
        "source": {
            "repo": REPO,
            "subset": "default",
            "split": "train",
            "revision": state["revision"],
            "license": "apache-2.0",
            "columns": ["text"],
            "note": "Hugging Face validation split is not included. Ids are source_file:row_group:row.",
        },
        "sizing": {
            "target_chars": int(state["target_chars"]),
            "matched_to": "data/fineweb2/rus_Cyrl/v001 totals.chars",
            "actual_may_exceed_target_by_one_document": True,
        },
        "sampling": {
            "seed": int(state["seed"]),
            "files_read": len(state["sources"]),
            "note": (
                "Train shards are shuffled with the seed. Rows are read in shard order "
                "until the character budget is met. The dataset's own validation split is ignored."
            ),
        },
        "filter": {
            "name": "strict_keep_no_language_score",
            "same_as": "scripts/build_fineweb2_rus.py strict_keep, without language_score",
            "cyr_ratio_min": CYR_MIN,
            "n_words_min": WORDS_MIN,
            "newline_ratio_max": NEWLINE_MAX,
            "dup_line_char_frac_max": DUP_LINE_MAX,
            "punct_line_frac_min": PUNCT_MIN,
            "language_score_min": None,
        },
        "totals": {
            "read_docs": read_docs,
            "kept_docs": kept_docs,
            "rejected_docs": int(state["rejected_docs"]),
            "keep_rate": (kept_docs / read_docs) if read_docs else 0.0,
            "chars": int(state["kept_chars"]),
            "words": int(state["kept_words"]),
        },
        "rejection_counts": state["fail_counts"],
        "length_quantiles_from_reservoir": {
            "reservoir_size": len(reservoir),
            "chars": quantiles(chars),
            "words": quantiles(words),
        },
        "load": f"pandas.read_parquet({load_path!r})",
        "columns": ["id", "text", "source_file", "row_group", "row"],
        "parts": state["parts"],
        "sources": state["sources"],
    }
    path = out_dir / "manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def build(out_dir: Path, target_chars: int, seed: int, revision: str | None, refresh: bool) -> None:
    manifest_path = out_dir / "manifest.json"
    state_path = out_dir / "_state.json"
    if refresh and out_dir.exists():
        import shutil

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
        state = new_state(revision, seed, target_chars, file_order, " ".join(sys.argv))
        save_state(state_path, state)
        print(f"revision {revision}  train files {len(file_order):,}", flush=True)
    else:
        if state["seed"] != seed or int(state["target_chars"]) != int(target_chars):
            raise SystemExit(
                "This output already has a different seed or target. Use --refresh or another --out directory."
            )
        revision = state["revision"]
        print(
            f"resume kept_docs={state['kept_docs']:,} chars={state['kept_chars']:,}/{target_chars:,}",
            flush=True,
        )

    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    known_parts = {part["file"] for part in state["parts"]}
    for path in parts_dir.glob("part-*.parquet"):
        if path.name not in known_parts:
            path.unlink()

    rng = rng_from_state(state, seed)
    reservoir: list[dict] = list(state.get("reservoir") or [])
    buffer = empty_buffer()
    completed = set(state["completed_files"])
    started = time.time()
    seen_kept = int(state["kept_docs"])

    def flush(source_file: str | None, next_row: int, file_read: int, file_kept: int) -> None:
        nonlocal buffer, seen_kept
        if buffer["id"]:
            part = write_part(parts_dir, len(state["parts"]), buffer)
            state["parts"].append(part)
            print(f"  wrote {part['file']} rows={part['rows']:,} bytes={part['bytes']:,}", flush=True)
            buffer = empty_buffer()
        state["reservoir"] = reservoir
        state["rng_state"] = rng_to_state(rng)
        state["cursor_file"] = source_file
        state["cursor_row"] = int(next_row)
        state["cursor_read_docs"] = int(file_read)
        state["cursor_kept_docs"] = int(file_kept)
        save_state(state_path, state)

    def fold_document(kept: bool, chars: int, words: int, fails: dict[str, int]) -> None:
        nonlocal seen_kept
        state["read_docs"] += 1
        for key, failed in fails.items():
            state["fail_counts"][key] += failed
        if not kept:
            state["rejected_docs"] += 1
            return
        state["kept_docs"] += 1
        state["kept_chars"] += chars
        state["kept_words"] += words
        seen_kept += 1
        reservoir_add(reservoir, {"n_chars": int(chars), "n_words": int(words)}, seen_kept, rng)
        if len(reservoir) > RESERVOIR_SIZE:
            raise RuntimeError("Reservoir grew past its cap.")

    for path in state["file_order"]:
        if int(state["kept_chars"]) >= target_chars:
            break
        if path in completed:
            continue
        print(f"download {path}", flush=True)
        local = download_shard(path, revision)
        parquet = pq.ParquetFile(local)
        if state.get("cursor_file") == path:
            next_row = int(state["cursor_row"])
            file_read = int(state["cursor_read_docs"])
            file_kept = int(state["cursor_kept_docs"])
        else:
            next_row = 0
            file_read = 0
            file_kept = 0
        row = 0
        stop = False
        for group_index in range(int(parquet.num_row_groups)):
            if stop:
                break
            for batch in parquet.iter_batches(batch_size=BATCH_ROWS, row_groups=[group_index], columns=["text"]):
                if stop:
                    break
                texts = batch.column("text").to_pylist()
                for text in texts:
                    if row < next_row:
                        row += 1
                        continue
                    kept, chars, words, fails = screen(text)
                    fold_document(kept, chars, words, fails)
                    file_read += 1
                    if kept and isinstance(text, str):
                        file_kept += 1
                        buffer["id"].append(f"{path}:{group_index}:{row}")
                        buffer["text"].append(text)
                        buffer["source_file"].append(path)
                        buffer["row_group"].append(group_index)
                        buffer["row"].append(row)
                    row += 1
                    if len(buffer["id"]) >= PART_ROWS:
                        flush(path, row, file_read, file_kept)
                    if int(state["kept_chars"]) >= target_chars:
                        stop = True
                        break
                if file_read and file_read % 20_000 < BATCH_ROWS:
                    elapsed = time.time() - started
                    done = int(state["kept_chars"])
                    print(
                        f"{path} row={row:,} kept_chars={done:,}/{target_chars:,} "
                        f"keep_rate={state['kept_docs'] / state['read_docs']:.3f} "
                        f"elapsed={elapsed / 60:.1f}min",
                        flush=True,
                    )
        flush(path, row, file_read, file_kept)
        if stop:
            state["sources"].append(
                {
                    "source_file": path,
                    "rows_read": row,
                    "rows_in_file": int(parquet.metadata.num_rows),
                    "read_docs": file_read,
                    "kept_docs": file_kept,
                    "partial": True,
                }
            )
            save_state(state_path, state)
            break
        state["sources"].append(
            {
                "source_file": path,
                "rows_read": row,
                "rows_in_file": int(parquet.metadata.num_rows),
                "read_docs": file_read,
                "kept_docs": file_kept,
                "partial": False,
            }
        )
        state["completed_files"].append(path)
        completed.add(path)
        state["cursor_file"] = None
        state["cursor_row"] = 0
        state["cursor_read_docs"] = 0
        state["cursor_kept_docs"] = 0
        save_state(state_path, state)
        print(f"finished {path} read={file_read:,} kept={file_kept:,}", flush=True)

    if int(state["kept_chars"]) < target_chars:
        raise SystemExit(f"Ran out of train shards at {int(state['kept_chars']):,} characters.")
    manifest = write_manifest(out_dir, state)
    totals = json.loads(manifest.read_text(encoding="utf-8"))["totals"]
    print(json.dumps({"manifest": rel(manifest), **totals}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-chars", type=int, default=TARGET_CHARS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--revision", default=None, help="Hugging Face dataset revision. Default: current main sha, then frozen in the build.")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    build(
        out_dir=args.out if args.out.is_absolute() else ROOT / args.out,
        target_chars=args.target_chars,
        seed=args.seed,
        revision=args.revision,
        refresh=args.refresh,
    )


if __name__ == "__main__":
    main()
