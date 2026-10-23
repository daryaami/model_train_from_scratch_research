"""Learning rate по номеру шага.

Модель про это расписание не знает. train.py спрашивает число и подставляет
его в оптимизатор. Два режима читают один и тот же блок конфига:

- wsd: линейный разогрев от 0 до пика, полка на пике, затем спад до min_lr.
  Доли warmup, stable и decay в сумме дают 1.
- cosine: тот же разогрев, затем cosine до min_lr до конца обучения, без полки.
  Доля stable в этом режиме не используется.

Поле decay выбирает форму спада внутри WSD: cosine или linear.
"""

from __future__ import annotations

import math


def learning_rate(step: int, total_steps: int, schedule: dict) -> float:
    """Learning rate на шаге step. Шаги нумеруются с нуля и идут до total_steps - 1."""
    if total_steps < 1:
        raise ValueError("total_steps должен быть не меньше 1.")
    if step < 0 or step >= total_steps:
        raise ValueError(f"Шаг {step} вне диапазона 0..{total_steps - 1}.")
    name = schedule["name"]
    peak = float(schedule["peak_lr"])
    min_lr = float(schedule["min_lr"])
    warmup_frac = float(schedule["warmup_frac"])
    if name == "wsd":
        warm, stable, decay = _wsd_lengths(total_steps, schedule)
        if warm > 0 and step < warm:
            return peak * step / warm
        if step < warm + stable:
            return peak
        return _decay(step - warm - stable, decay, peak, min_lr, schedule["decay"])
    if name == "cosine":
        warm = int(total_steps * warmup_frac)
        if warm > 0 and step < warm:
            return peak * step / warm
        return _decay(step - warm, total_steps - warm, peak, min_lr, "cosine")
    raise ValueError(f"Неизвестное расписание {name!r}. Ожидается wsd или cosine.")


def validate_schedule(schedule: dict) -> None:
    """Поймать опечатку в долях и имени до старта обучения."""
    name = schedule.get("name")
    if name not in ("wsd", "cosine"):
        raise ValueError(f"Неизвестное расписание {name!r}. Ожидается wsd или cosine.")
    decay = schedule.get("decay")
    if decay not in ("cosine", "linear"):
        raise ValueError(f"Неизвестный спад {decay!r}. Ожидается cosine или linear.")
    total = float(schedule["warmup_frac"]) + float(schedule["stable_frac"]) + float(schedule["decay_frac"])
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Доли warmup, stable и decay дают {total}, а нужна 1.")
    if float(schedule["peak_lr"]) < float(schedule["min_lr"]):
        raise ValueError("peak_lr меньше min_lr.")


def self_check() -> None:
    """Коротко проверить оба режима на фиксированных долях, не на числах из yaml.

    На шаге 0 WSD равен нулю, на полке — пику, в конце — min_lr.
    Там, где у WSD ещё полка, cosine уже снижается. linear и cosine в середине
    спада не совпадают.
    """
    demo = {
        "name": "wsd",
        "peak_lr": 1.0,
        "min_lr": 0.1,
        "warmup_frac": 0.1,
        "stable_frac": 0.7,
        "decay_frac": 0.2,
        "decay": "cosine",
    }
    validate_schedule(demo)
    total = 100
    warm = int(total * demo["warmup_frac"])
    if learning_rate(0, total, demo) != 0.0:
        raise RuntimeError("WSD на шаге 0 должен быть 0.")
    if learning_rate(warm, total, demo) != 1.0:
        raise RuntimeError("WSD в начале полки должен быть равен пику.")
    if abs(learning_rate(total - 1, total, demo) - 0.1) > 1e-8:
        raise RuntimeError("WSD на последнем шаге должен дойти до min_lr.")
    cosine = dict(demo, name="cosine")
    mid_shelf = warm + int(total * demo["stable_frac"]) // 2
    if learning_rate(mid_shelf, total, demo) != 1.0:
        raise RuntimeError("Полка WSD не держит пик.")
    cosine_mid = learning_rate(mid_shelf, total, cosine)
    if not 0.1 < cosine_mid < 1.0:
        raise RuntimeError("Cosine на месте полки WSD уже должен снижаться.")
    decay_start = warm + int(total * demo["stable_frac"])
    decay_mid = decay_start + (total - decay_start) // 2
    cosine_decay = learning_rate(decay_mid, total, demo)
    linear_decay = learning_rate(decay_mid, total, dict(demo, decay="linear"))
    if abs(cosine_decay - linear_decay) < 1e-6:
        raise RuntimeError("cosine и linear спад совпали там, где должны различаться.")
    try:
        learning_rate(decay_mid, total, dict(demo, decay="step"))
    except ValueError:
        return
    raise RuntimeError("Неизвестный тип спада не был отвергнут.")


def _wsd_lengths(total_steps: int, schedule: dict) -> tuple[int, int, int]:
    warm = int(total_steps * float(schedule["warmup_frac"]))
    stable = int(total_steps * float(schedule["stable_frac"]))
    if warm + stable > total_steps:
        stable = total_steps - warm
    decay = total_steps - warm - stable
    return warm, stable, decay


def _decay(index: int, length: int, peak: float, min_lr: float, kind: str) -> float:
    """index 0 — ещё пик, последний шаг отрезка — уже min_lr."""
    if length <= 1:
        progress = 1.0
    else:
        progress = index / (length - 1)
    progress = min(max(progress, 0.0), 1.0)
    if kind == "cosine":
        # От 1 до 0 по половине косинуса: сначала медленно, в середине быстрее, в конце снова медленно.
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    elif kind == "linear":
        coeff = 1.0 - progress
    else:
        raise ValueError(f"Неизвестный спад {kind!r}. Ожидается cosine или linear.")
    return min_lr + (peak - min_lr) * coeff
