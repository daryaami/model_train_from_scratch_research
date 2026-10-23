"""Цикл обучения: батч, loss, шаг оптимизатора, расписание.

Архитектура здесь не выбирается заново. train.py только двигает веса
LanguageModel и спрашивает learning rate у schedule.py.

Каждый запуск создаёт свою папку внутри paths.checkpoint_dir и ничего
в соседних прогонах не удаляет. --resume продолжает папку того чекпоинта,
который ему передали. Внутри папки:
- train_log.jsonl — строка на каждый шаг;
- generations.jsonl — жадные продолжения на шагах validation и лист в конце;
- step_XXXXXXX.pt — полный чекпоинт раз в save_interval и на последнем шаге;
- last.pt — копия самого свежего из них, с него продолжают --resume;
- best.pt — шаг с наименьшим промежуточным validation loss;
- summary.json — полный validation, один test, время и память. Пишется в конце.

Проверка готовности этапа 2, без длинного обучения:

    python -m lm.train --check
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from lm.config import ROOT, load_config, project_path
from lm.data import TokenStream
from lm.evaluate import validation_loss
from lm.generate import generate
from lm.model import LanguageModel, ModelConfig, count_parameters
from lm.schedule import learning_rate, self_check as schedule_self_check
from lm.schedule import validate_schedule
from tokenizers import Tokenizer

CHECK_PROMPT = "Однажды в"
CHECK_BATCH = 2
CHECK_NEW_TOKENS = 16


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(cfg: dict, device: torch.device) -> LanguageModel:
    model = LanguageModel(ModelConfig.from_dict(cfg["model"]))
    return model.to(device)


def open_tokenizer(cfg: dict) -> Tokenizer:
    path = project_path(cfg["paths"]["tokenizer"]) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(path))
    expected = int(cfg["model"]["vocab_size"])
    if tokenizer.get_vocab_size() != expected:
        raise SystemExit(f"В конфиге словарь {expected}, у токенизатора {tokenizer.get_vocab_size()}.")
    return tokenizer


def param_groups(model: LanguageModel, weight_decay: float) -> list[dict]:
    """Decay только на матрицах. Нормы и одномерные векторы не гниют к нулю.

    Иначе RMSNorm, у которой и так один множитель на канал, сдувается weight decay.
    """
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for param in model.parameters():
        if param.data_ptr() in seen or not param.requires_grad:
            continue
        seen.add(param.data_ptr())
        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def tokens_per_step(train_cfg: dict, context: int) -> int:
    return int(train_cfg["batch_size"]) * int(train_cfg["grad_accum"]) * context


def peak_mem_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def prompts_from(cfg: dict) -> list[str]:
    raw = (cfg.get("eval") or {}).get("prompts")
    if not raw:
        return ["Однажды в"]
    if not isinstance(raw, list) or not all(isinstance(item, str) and item for item in raw):
        raise SystemExit("eval.prompts должен быть списком непустых строк.")
    return raw


def cyrillic_frac(text: str) -> float | None:
    """Доля кириллицы среди букв. None, если в хвосте нет ни одной буквы."""
    letters = 0
    cyrillic = 0
    for char in text:
        if not char.isalpha():
            continue
        letters += 1
        if "\u0400" <= char <= "\u04FF":
            cyrillic += 1
    if letters == 0:
        return None
    return cyrillic / letters


def unique_frac(ids: tuple[int, ...]) -> float | None:
    if not ids:
        return None
    return len(set(ids)) / len(ids)


def allocate_run_dir(root: Path, label: str) -> Path:
    """Новая папка прогона. Уже лежащие рядом каталоги не трогаем."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in label.strip())
    if not safe:
        safe = "run"
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / f"{stamp}_{safe}"
    suffix = 2
    while candidate.exists():
        candidate = root / f"{stamp}_{safe}_{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


def run_dir_name(path: Path) -> str:
    """Путь в summary: относительно корня репозитория, если папка внутри него."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def best_from_log(log_path: Path) -> tuple[float, int | None]:
    """При --resume не затирать best.pt шагом хуже уже записанного."""
    best_loss = math.inf
    best_step = None
    if not log_path.exists():
        return best_loss, best_step
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        value = row.get("val_loss")
        if isinstance(value, (int, float)) and math.isfinite(value) and value < best_loss:
            best_loss = float(value)
            best_step = int(row["step"])
    return best_loss, best_step


def generation_record(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    step: int,
    phase: str,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> dict:
    sample = generate(model, tokenizer, prompt, max_new_tokens, temperature, device, generator)
    return {
        "step": step,
        "phase": phase,
        "prompt": prompt,
        "continuation": sample.continuation,
        "new_tokens": sample.new_tokens,
        "temperature": temperature,
        "stopped_on_eos": sample.stopped_on_eos,
        "cyrillic_frac": cyrillic_frac(sample.continuation),
        "unique_frac": unique_frac(sample.new_ids),
    }


def write_generations(
    path: Path,
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompts: list[str],
    step: int,
    phase: str,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> None:
    for prompt in prompts:
        append_jsonl(
            path,
            generation_record(
                model,
                tokenizer,
                prompt,
                step,
                phase,
                max_new_tokens,
                temperature,
                device,
                generator,
            ),
        )


def open_split(cfg: dict, name: str, block: int, seed: int) -> TokenStream:
    path = project_path(cfg["paths"]["dataset"]) / "tokens" / f"{name}.bin"
    if not path.exists():
        raise FileNotFoundError(path)
    return TokenStream(path, block, seed)


def save_checkpoint(
    path: Path,
    model: LanguageModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: dict,
) -> None:
    """Атомарная запись: оборванный прогон не оставляет половину файла."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": {key: value for key, value in cfg.items() if not key.startswith("_")},
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(cfg: dict, resume: Path | None, name: str | None = None) -> None:
    """Основной цикл. На каждом шаге learning rate берётся из schedule.name."""
    validate_schedule(cfg["schedule"])
    seed = int(cfg["train"]["seed"])
    set_seed(seed)
    device = pick_device()
    model = build_model(cfg, device)
    train_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(
        param_groups(model, float(train_cfg["weight_decay"])),
        lr=float(cfg["schedule"]["peak_lr"]),
        betas=tuple(train_cfg["betas"]),
    )
    start = 0
    if resume is not None:
        payload = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start = int(payload["step"]) + 1
    block = int(cfg["model"]["context"])
    stream = open_split(cfg, "train", block, seed)
    val_stream = open_split(cfg, "validation", block, seed)
    max_steps = int(train_cfg["max_steps"])
    grad_accum = int(train_cfg["grad_accum"])
    batch_size = int(train_cfg["batch_size"])
    grad_clip = float(train_cfg["grad_clip"])
    per_step = tokens_per_step(train_cfg, block)
    eval_interval = int(train_cfg["eval_interval"])
    eval_batches = int(train_cfg["eval_batches"])
    save_interval = int(train_cfg["save_interval"])
    log_interval = int(train_cfg["log_interval"])
    gen_new_tokens = int(train_cfg.get("gen_new_tokens", 64))
    final_new_tokens = int(train_cfg.get("final_new_tokens", 128))
    sample_temperature = float(train_cfg.get("sample_temperature", 0.8))
    prompts = prompts_from(cfg)
    tokenizer = open_tokenizer(cfg)
    for prompt in prompts:
        ids = tokenizer.encode(prompt).ids
        if not ids or len(ids) > block:
            raise SystemExit(f"Промпт {prompt!r} пустой или длиннее окна {block}.")
    if resume is not None:
        checkpoint_dir = resume.resolve().parent
    else:
        label = name or str(cfg["schedule"]["name"])
        checkpoint_dir = allocate_run_dir(project_path(cfg["paths"]["checkpoint_dir"]), label)
    log_path = checkpoint_dir / "train_log.jsonl"
    generations_path = checkpoint_dir / "generations.jsonl"
    best_loss, best_step = best_from_log(log_path) if resume is not None else (math.inf, None)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(
        f"run_dir={run_dir_name(checkpoint_dir)} device={device} parameters={count_parameters(model):,} "
        f"intermediate={model.config.intermediate_size} steps={start}..{max_steps - 1}",
        flush=True,
    )
    started = time.perf_counter()
    model.train()
    for step in range(start, max_steps):
        step_started = time.perf_counter()
        lr = learning_rate(step, max_steps, cfg["schedule"])
        set_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(grad_accum):
            inputs, targets = stream.random_batch(batch_size)
            _, loss = model(inputs.to(device), targets.to(device))
            (loss / grad_accum).backward()
            running += float(loss.item())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
        optimizer.step()
        step_time = time.perf_counter() - step_started
        record = {
            "step": step,
            "tokens_seen": (step + 1) * per_step,
            "train_loss": running / grad_accum,
            "lr": lr,
            "grad_norm": grad_norm,
            "step_time_s": step_time,
            "peak_mem_mb": peak_mem_mb(device),
        }
        evaluate_now = step % eval_interval == 0 or step == max_steps - 1
        if evaluate_now:
            eval_started = time.perf_counter()
            stats = validation_loss(model, val_stream, batch_size, eval_batches, device)
            record["val_loss"] = stats.loss
            record["val_tokens"] = stats.tokens
            record["eval_time_s"] = time.perf_counter() - eval_started
            write_generations(
                generations_path,
                model,
                tokenizer,
                prompts,
                step,
                "train",
                gen_new_tokens,
                0.0,
                device,
            )
            model.train()
            if math.isfinite(stats.loss) and stats.loss < best_loss:
                best_loss = stats.loss
                best_step = step
                save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, step, cfg)
        if "val_loss" in record or step % log_interval == 0 or step == max_steps - 1:
            print(json.dumps(record, ensure_ascii=False), flush=True)
        append_jsonl(log_path, record)
        if step % save_interval == 0 or step == max_steps - 1:
            save_checkpoint(checkpoint_dir / f"step_{step:07d}.pt", model, optimizer, step, cfg)
            save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, step, cfg)
    train_time = time.perf_counter() - started
    if best_step is None:
        raise RuntimeError("Не было ни одного validation, best.pt не из чего выбрать.")
    print(f"full eval of best.pt step={best_step}", flush=True)
    load_best = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(load_best["model"])
    full_val = validation_loss(model, val_stream, batch_size, None, device, log_every=100)
    test_stream = open_split(cfg, "test", block, seed)
    test_stats = validation_loss(model, test_stream, batch_size, None, device, log_every=100)
    sample_generator = torch.Generator(device=device)
    sample_generator.manual_seed(seed)
    for prompt in prompts:
        append_jsonl(
            generations_path,
            generation_record(
                model, tokenizer, prompt, best_step, "final", final_new_tokens, 0.0, device
            ),
        )
        append_jsonl(
            generations_path,
            generation_record(
                model,
                tokenizer,
                prompt,
                best_step,
                "final",
                final_new_tokens,
                sample_temperature,
                device,
                sample_generator,
            ),
        )
    summary = {
        "best_step": best_step,
        "best_probe_val_loss": best_loss,
        "full_val_loss": full_val.loss,
        "full_val_tokens": full_val.tokens,
        "test_loss": test_stats.loss,
        "test_tokens": test_stats.tokens,
        "tokens_seen": max_steps * per_step,
        "tokens_per_step": per_step,
        "parameters": count_parameters(model),
        "train_time_s": train_time,
        "total_time_s": time.perf_counter() - started,
        "peak_mem_mb": peak_mem_mb(device),
        "weights": "best.pt",
        "run_dir": run_dir_name(checkpoint_dir),
        "max_steps": max_steps,
        "finished_step": max_steps - 1,
    }
    write_json(checkpoint_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def run_check(cfg: dict) -> None:
    """Один шаг и короткая генерация. Это критерий этапа 2, не дымовой прогон этапа 3.

    Learning rate на шаге 0 у WSD равен нулю, и веса бы не сдвинулись.
    Поэтому сам шаг проверки берёт пиковый lr. Что расписание в нуле
    действительно нулевое, отдельно проверяет schedule.self_check.
    """
    validate_schedule(cfg["schedule"])
    schedule_self_check()
    seed = int(cfg["train"]["seed"])
    set_seed(seed)
    device = pick_device()
    model = build_model(cfg, device)
    n_params = count_parameters(model)
    print(
        f"parameters={n_params} intermediate={model.config.intermediate_size} "
        f"context={model.config.context} device={device}",
        flush=True,
    )
    block = int(cfg["model"]["context"])
    stream = TokenStream(project_path(cfg["paths"]["dataset"]) / "tokens" / "train.bin", block, seed)
    val_stream = TokenStream(project_path(cfg["paths"]["dataset"]) / "tokens" / "validation.bin", block, seed)
    inputs, targets = stream.random_batch(CHECK_BATCH)
    inputs = inputs.to(device)
    targets = targets.to(device)
    model.train()
    _, loss = model(inputs, targets)
    loss_value = float(loss.item())
    if not math.isfinite(loss_value) or loss_value > 100:
        raise RuntimeError(f"Loss после forward не выглядит конечным: {loss_value}")
    before = model.embed.weight.detach().clone()
    optimizer = torch.optim.AdamW(
        param_groups(model, float(cfg["train"]["weight_decay"])),
        lr=float(cfg["schedule"]["peak_lr"]),
        betas=tuple(cfg["train"]["betas"]),
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
    optimizer.step()
    delta = float((model.embed.weight.detach() - before).abs().max().item())
    if delta == 0.0:
        raise RuntimeError("После шага оптимизатора веса эмбеддинга не изменились.")
    val_loss = validation_loss(model, val_stream, CHECK_BATCH, 1, device).loss
    if not math.isfinite(val_loss):
        raise RuntimeError(f"Validation loss не конечный: {val_loss}")
    tokenizer = open_tokenizer(cfg)
    text = generate(
        model,
        tokenizer,
        CHECK_PROMPT,
        max_new_tokens=CHECK_NEW_TOKENS,
        temperature=0.8,
        device=device,
    ).text
    if not isinstance(text, str) or CHECK_PROMPT not in text:
        raise RuntimeError(f"Генерация не вернулась к исходному промпту: {text!r}")
    print(
        json.dumps(
            {
                "loss": loss_value,
                "val_loss": val_loss,
                "embed_max_abs_delta": delta,
                "generation": text,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--check", action="store_true", help="Один шаг и короткая генерация, без длинного обучения.")
    parser.add_argument("--resume", type=Path, default=None, help="Чекпоинт, с которого продолжить. Запись идёт в его папку.")
    parser.add_argument("--name", default=None, help="Подпись новой папки прогона. Без неё берётся имя расписания.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.check:
        run_check(cfg)
        return
    train(cfg, args.resume, args.name)


if __name__ == "__main__":
    main()
