"""Оценка базовой модели на русском: RuBLiMP и LanguageTool.

RuBLiMP — пары фраз. source_sentence грамматичная, target_sentence нет.
Пара засчитана, если средняя кросс-энтропия по токенам нашего BPE ниже
у грамматичной фразы. Это та же величина, что перплексия в статье RuBLiMP.
Случайный уровень 50%.

LanguageTool считает срабатывания русского набора правил на хвостах
из generations.jsonl рядом с чекпоинтом. Долю уникальных токенов модуль
не пересчитывает.

    python -m lm.russian_eval --checkpoint checkpoints/<прогон>/best.pt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import language_tool_python
import torch
import torch.nn.functional as F
from datasets import DatasetDict, get_dataset_config_names, load_dataset
from tokenizers import Tokenizer

from lm.config import project_path
from lm.model import LanguageModel, ModelConfig

RUBLIMP = "RussianNLP/rublimp"
BATCH = 128


def checkpoint_config(path: Path, device: torch.device) -> tuple[LanguageModel, dict]:
    """Веса и конфиг из одного чекпоинта. Оптимизатор не нужен."""
    model_payload = torch.load(path, map_location=device, weights_only=False)
    cfg = model_payload["config"]
    model = LanguageModel(ModelConfig.from_dict(cfg["model"])).to(device)
    model.load_state_dict(model_payload["model"])
    model.eval()
    return model, cfg


def encode_sentence(tokenizer: Tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text).ids)


@torch.no_grad()
def mean_token_nll(
    model: LanguageModel,
    sentences: list[list[int]],
    device: torch.device,
) -> list[float]:
    """Средняя кросс-энтропия по позициям, которые модель умеет предсказывать.

    logits[t] предсказывает токен t+1, поэтому у фразы длины T в среднее
    входят T-1 чисел. Паддинга нет: id 0 — это EOS, и в один батч попадают
    только фразы одной длины.
    """
    scores = [0.0] * len(sentences)
    order = sorted(range(len(sentences)), key=lambda index: len(sentences[index]))
    cursor = 0
    while cursor < len(order):
        length = len(sentences[order[cursor]])
        end = cursor
        while end < len(order) and len(sentences[order[end]]) == length:
            end += 1
        group = order[cursor:end]
        for start in range(0, len(group), BATCH):
            chunk = group[start : start + BATCH]
            idx = torch.tensor([sentences[index] for index in chunk], dtype=torch.long, device=device)
            logits, _ = model(idx)
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            targets = idx[:, 1:].unsqueeze(-1)
            nll = -log_probs.gather(-1, targets).squeeze(-1)
            for index, value in zip(chunk, nll.mean(dim=1).tolist()):
                scores[index] = float(value)
        cursor = end
    return scores


def empty_bucket() -> dict[str, int]:
    return {"n_correct": 0, "n_scored": 0}


def add_pair(bucket: dict[str, int], correct: bool) -> None:
    bucket["n_scored"] += 1
    if correct:
        bucket["n_correct"] += 1


def with_accuracy(bucket: dict[str, int]) -> dict[str, float | int]:
    scored = bucket["n_scored"]
    accuracy = bucket["n_correct"] / scored if scored else 0.0
    return {"accuracy": accuracy, "n_correct": bucket["n_correct"], "n_scored": scored}


def score_rublimp(
    model: LanguageModel,
    tokenizer: Tokenizer,
    device: torch.device,
) -> dict:
    context = model.config.context
    names = get_dataset_config_names(RUBLIMP)
    kept_source: list[list[int]] = []
    kept_target: list[list[int]] = []
    kept_level: list[str] = []
    kept_phenomenon: list[str] = []
    n_skipped = 0
    for name in names:
        dataset = load_dataset(RUBLIMP, name)
        if isinstance(dataset, DatasetDict):
            rows = [row for split in dataset.values() for row in split]
        else:
            rows = list(dataset)
        before = len(kept_source)
        skipped_here = 0
        for row in rows:
            source = encode_sentence(tokenizer, row["source_sentence"])
            target = encode_sentence(tokenizer, row["target_sentence"])
            if len(source) < 2 or len(target) < 2 or len(source) > context or len(target) > context:
                n_skipped += 1
                skipped_here += 1
                continue
            kept_source.append(source)
            kept_target.append(target)
            kept_level.append(str(row["level"]))
            kept_phenomenon.append(str(row["phenomenon"]))
        print(
            f"rublimp {name}: pairs={len(rows)} kept={len(kept_source) - before} skipped={skipped_here}",
            flush=True,
        )
    print(f"rublimp scoring {len(kept_source)} pairs", flush=True)
    source_nll = mean_token_nll(model, kept_source, device)
    target_nll = mean_token_nll(model, kept_target, device)
    overall = empty_bucket()
    by_level: dict[str, dict[str, int]] = defaultdict(empty_bucket)
    by_phenomenon: dict[str, dict[str, int]] = defaultdict(empty_bucket)
    for source, target, level, phenomenon in zip(source_nll, target_nll, kept_level, kept_phenomenon):
        correct = source < target
        add_pair(overall, correct)
        add_pair(by_level[level], correct)
        add_pair(by_phenomenon[phenomenon], correct)
    return {
        "dataset": RUBLIMP,
        "n_configs": len(names),
        "accuracy": with_accuracy(overall)["accuracy"],
        "n_correct": overall["n_correct"],
        "n_scored": overall["n_scored"],
        "n_skipped": n_skipped,
        "by_level": {name: with_accuracy(bucket) for name, bucket in sorted(by_level.items())},
        "by_phenomenon": {name: with_accuracy(bucket) for name, bucket in sorted(by_phenomenon.items())},
    }


def score_languagetool(generations: Path, tool) -> dict:
    """Срабатывания уже запущенного LanguageTool по полю continuation."""
    n_matches = 0
    n_continuations = 0
    with generations.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            n_matches += len(tool.check(record["continuation"]))
            n_continuations += 1
    return {
        "language": "ru",
        "n_matches": n_matches,
        "n_continuations": n_continuations,
        "generations": str(generations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = checkpoint_config(checkpoint, device)
    tokenizer_dir = project_path(cfg["paths"]["tokenizer"])
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    generations = checkpoint.parent / "generations.jsonl"
    if not generations.is_file():
        raise SystemExit(f"Рядом с чекпоинтом нет {generations}.")
    # Сервер поднимается до RuBLiMP: первый вызов скачивает правила и сразу
    # показывает, что Java на месте. Пока идёт счёт пар, процесс просто ждёт.
    print("languagetool starting", flush=True)
    tool = language_tool_python.LanguageTool("ru")
    print("languagetool ready", flush=True)
    try:
        report = {
            "checkpoint": str(checkpoint),
            "tokenizer": str(tokenizer_dir),
            "rublimp": score_rublimp(model, tokenizer, device),
            "languagetool": score_languagetool(generations, tool),
        }
    finally:
        tool.close()
    destination = checkpoint.parent / "russian_eval.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    rublimp = report["rublimp"]
    tool = report["languagetool"]
    print(
        f"rublimp accuracy={rublimp['accuracy']:.4f} "
        f"scored={rublimp['n_scored']} skipped={rublimp['n_skipped']}",
        flush=True,
    )
    print(
        f"languagetool matches={tool['n_matches']} continuations={tool['n_continuations']}",
        flush=True,
    )
    print(destination, flush=True)


if __name__ == "__main__":
    main()
