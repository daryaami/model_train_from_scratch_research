"""Дописать текст токен за токеном.

Промпт кодируется тем же BPE 16 000, что и корпус. На каждом шаге модель
видит не больше context последних токенов, берётся распределение по словарю
и из него выбирается один id. id 0 — EOS, на нём генерация останавливается.
Кэш внимания не используется: для короткой пробы дешевле пересчитать окно.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import torch
from tokenizers import Tokenizer

from lm.config import load_config, project_path
from lm.model import LanguageModel, ModelConfig, load_weights

EOS_TOKEN = "<|endoftext|>"


class Generation(NamedTuple):
    """Полный текст и хвост, который модель дописала сама."""

    text: str
    continuation: str
    new_tokens: int
    new_ids: tuple[int, ...]
    stopped_on_eos: bool


@torch.no_grad()
def generate(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Generation:
    """Дописать промпт. Хвостовой EOS в текст не входит.

    temperature == 0 берёт самый вероятный id и генератор случайных чисел не трогает.
    Иначе id выбирается из распределения, и generator фиксирует этот выбор.
    """
    was_training = model.training
    model.eval()
    prompt_ids = tokenizer.encode(prompt).ids
    context = model.config.context
    if len(prompt_ids) > context:
        prompt_ids = prompt_ids[-context:]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    produced: list[int] = []
    for _ in range(max_new_tokens):
        window = idx[:, -context:]
        logits, _ = model(window)
        next_logits = logits[:, -1, :]
        if temperature == 0:
            next_id = int(torch.argmax(next_logits, dim=-1).item())
        else:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
        produced.append(next_id)
        if next_id == eos_id:
            break
        idx = torch.cat([idx, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
    if was_training:
        model.train()
    stopped = bool(produced) and produced[-1] == eos_id
    full_ids = prompt_ids + produced
    if full_ids and full_ids[-1] == eos_id:
        full_ids = full_ids[:-1]
    text = tokenizer.decode(full_ids)
    prompt_text = tokenizer.decode(prompt_ids)
    if text.startswith(prompt_text):
        continuation = text[len(prompt_text) :]
    else:
        continuation = text
    return Generation(text, continuation, len(produced), tuple(produced), stopped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LanguageModel(ModelConfig.from_dict(cfg["model"])).to(device)
    load_weights(model, args.checkpoint, device)
    tokenizer = Tokenizer.from_file(str(project_path(cfg["paths"]["tokenizer"]) / "tokenizer.json"))
    text = generate(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, device).text
    print(text)


if __name__ == "__main__":
    main()
