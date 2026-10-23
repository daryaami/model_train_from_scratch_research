"""Средний loss по отложенному потоку, без обновления весов.

Окна идут по файлу подряд, поэтому повторный запуск на тех же max_batches
даёт тот же набор примеров. Среднее считается по токенам, а не по батчам:
короткий последний батч не весит столько же, сколько полный.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import torch

from lm.config import load_config, project_path
from lm.data import TokenStream
from lm.model import LanguageModel, ModelConfig, load_weights


class LossStats(NamedTuple):
    """loss — среднее cross-entropy на токен, tokens — сколько целей в него вошло."""

    loss: float
    tokens: int
    batches: int


@torch.no_grad()
def validation_loss(
    model: LanguageModel,
    stream: TokenStream,
    batch_size: int,
    max_batches: int | None,
    device: torch.device,
    log_every: int | None = None,
) -> LossStats:
    """Среднее cross-entropy. max_batches=None проходит файл до конца."""
    was_training = model.training
    model.eval()
    total = 0.0
    tokens = 0
    batches = 0
    for inputs, targets in stream.sequential_batches(batch_size, max_batches):
        _, loss = model(inputs.to(device), targets.to(device))
        count = int(inputs.numel())
        total += float(loss.item()) * count
        tokens += count
        batches += 1
        if log_every is not None and batches % log_every == 0:
            print(f"eval_batches={batches} tokens={tokens} loss={total / tokens:.4f}", flush=True)
    if was_training:
        model.train()
    if tokens == 0:
        raise RuntimeError("Оценка не дала ни одного батча.")
    return LossStats(total / tokens, tokens, batches)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--full", action="store_true", help="Весь файл сплита, не только eval_batches.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LanguageModel(ModelConfig.from_dict(cfg["model"])).to(device)
    load_weights(model, args.checkpoint, device)
    block = int(cfg["model"]["context"])
    stream = TokenStream(
        project_path(cfg["paths"]["dataset"]) / "tokens" / f"{args.split}.bin",
        block,
        int(cfg["train"]["seed"]),
    )
    if args.full:
        limit = None
    elif args.max_batches is not None:
        limit = args.max_batches
    else:
        limit = int(cfg["train"]["eval_batches"])
    stats = validation_loss(
        model,
        stream,
        int(cfg["train"]["batch_size"]),
        limit,
        device,
        log_every=100 if args.full else None,
    )
    print(
        f"split={args.split} loss={stats.loss:.4f} tokens={stats.tokens} "
        f"batches={stats.batches} device={device}"
    )


if __name__ == "__main__":
    main()
