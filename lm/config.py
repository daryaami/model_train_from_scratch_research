"""Чтение configs/baseline.yaml.

Один файл описывает и модель, и обучение. Смена расписания — поле
schedule.name, архитектуру для этого менять не нужно.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    """Прочитать yaml. Относительный путь считается от корня репозитория."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"В {config_path} нет словаря настроек.")
    loaded["_config_path"] = str(config_path)
    return loaded


def project_path(value: str) -> Path:
    """Путь из конфига: абсолютный оставляем, относительный клеим к корню репозитория."""
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path
