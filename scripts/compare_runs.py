#!/usr/bin/env python3
"""Write vs_fineweb.json for a Cultura run against the FineWeb 10.9M pilot.

Loss on each run's own validation is not a shared test: the held-out text
comes from that run's corpus. RuBLiMP, LanguageTool, and the two cross
losses are the shared comparison. Cross losses are the numbers printed by
lm.evaluate --full on the other corpus's validation split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fertility(dataset_manifest: Path) -> dict:
    manifest = load_json(dataset_manifest)
    splits = {}
    for name, item in manifest["splits"].items():
        splits[name] = {
            "docs": item["docs"],
            "tokens": item["tokens"],
            "content_tokens": item["content_tokens"],
            "chars": item["chars"],
            "chars_per_content_token": item["chars_per_content_token"],
        }
    return splits


def run_record(checkpoint_dir: Path, dataset_manifest: Path) -> dict:
    summary = load_json(checkpoint_dir / "summary.json")
    russian = load_json(checkpoint_dir / "russian_eval.json")
    rublimp = russian["rublimp"]
    tool = russian["languagetool"]
    levels = {
        name: {"accuracy": bucket["accuracy"], "n_scored": bucket["n_scored"]}
        for name, bucket in rublimp["by_level"].items()
    }
    return {
        "run_dir": summary["run_dir"],
        "parameters": summary["parameters"],
        "max_steps": summary["max_steps"],
        "tokens_seen": summary["tokens_seen"],
        "full_val_loss": summary["full_val_loss"],
        "test_loss": summary["test_loss"],
        "rublimp_accuracy": rublimp["accuracy"],
        "rublimp_n_scored": rublimp["n_scored"],
        "rublimp_by_level": levels,
        "languagetool_matches": tool["n_matches"],
        "languagetool_continuations": tool["n_continuations"],
        "fertility": fertility(dataset_manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cultura-run", type=Path, required=True)
    parser.add_argument("--fineweb-run", type=Path, required=True)
    parser.add_argument("--cultura-dataset", type=Path, required=True)
    parser.add_argument("--fineweb-dataset", type=Path, required=True)
    parser.add_argument("--cultura-on-fineweb-val", type=float, required=True)
    parser.add_argument("--fineweb-on-cultura-val", type=float, required=True)
    parser.add_argument("--out", type=Path, default=None, help="Default: <cultura-run>/vs_fineweb.json")
    args = parser.parse_args()
    cultura = run_record(args.cultura_run, args.cultura_dataset)
    fineweb = run_record(args.fineweb_run, args.fineweb_dataset)
    cultura_train = cultura["fertility"]["train"]["chars_per_content_token"]
    fineweb_train = fineweb["fertility"]["train"]["chars_per_content_token"]
    report = {
        "note": (
            "Own validation and test losses are on each corpus's held-out split. "
            "RuBLiMP, LanguageTool, and the cross losses use the same tokenizer and the same 10.9M architecture."
        ),
        "fineweb": fineweb,
        "cultura": cultura,
        "cross_val_loss": {
            "cultura_weights_on_fineweb_validation": args.cultura_on_fineweb_val,
            "fineweb_weights_on_cultura_validation": args.fineweb_on_cultura_val,
        },
        "chars_per_content_token_train": {
            "fineweb": fineweb_train,
            "cultura": cultura_train,
            "ratio_cultura_over_fineweb": (cultura_train / fineweb_train) if fineweb_train else None,
        },
    }
    destination = args.out or (args.cultura_run / "vs_fineweb.json")
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(destination)


if __name__ == "__main__":
    main()
