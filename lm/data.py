"""Окна для обучения из потока uint16.

Файлы dataset/tokens/*.bin — это документы подряд, после каждого стоит EOS (id 0).
Ноль здесь настоящий токен конца документа, а не пустая ячейка и не паддинг.
Длинный документ не обрезается: он просто продолжается в следующих окнах,
потому что граница документа уже записана как EOS внутри потока.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class TokenStream:
    """Случайные окна для train и подряд идущие окна для validation.

    На одно окно нужно block_size токенов входа и ещё один токен справа:
    цель в позиции t — это токен t+1. Хвост всего файла короче 512 токенов
    в окна не попадает. Это меньше одного окна на сотни миллионов токенов.
    """

    def __init__(self, path: Path, block_size: int, seed: int) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        n_bytes = path.stat().st_size
        if n_bytes % 2 != 0:
            raise RuntimeError(f"{path} имеет нечётный размер, это не поток uint16.")
        n_tokens = n_bytes // 2
        if n_tokens <= block_size:
            raise RuntimeError(f"{path} короче одного окна в {block_size} токенов.")
        # little-endian, как записал prepare_dataset.py. В память файл не читается целиком.
        self.data = np.memmap(path, dtype="<u2", mode="r", shape=(n_tokens,))
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def _batch(self, starts: np.ndarray | list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        block = self.block_size
        inputs = np.stack([np.asarray(self.data[start : start + block], dtype=np.int64) for start in starts])
        targets = np.stack(
            [np.asarray(self.data[start + 1 : start + 1 + block], dtype=np.int64) for start in starts]
        )
        return torch.from_numpy(inputs), torch.from_numpy(targets)

    def random_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Случайные места в train. Генератор свой, с сидом из конфига, и не зависит от torch."""
        # Стартовый индекс + окно + один токен цели должны уместиться в файл.
        high = len(self.data) - self.block_size
        starts = self.rng.integers(0, high, size=batch_size)
        return self._batch(starts)

    def sequential_batches(self, batch_size: int, max_batches: int | None = None):
        """Окна validation подряд, без перемешивания, чтобы loss можно было повторить.

        Шаг равен длине окна: соседние батчи не смотрят на одни и те же токены
        как на вход. Цель последнего токена окна — это первый токен следующего.
        """
        block = self.block_size
        max_start = len(self.data) - block - 1
        produced = 0
        bucket: list[int] = []
        for start in range(0, max_start + 1, block):
            bucket.append(start)
            if len(bucket) < batch_size:
                continue
            yield self._batch(bucket)
            bucket = []
            produced += 1
            if max_batches is not None and produced >= max_batches:
                return
        if bucket and (max_batches is None or produced < max_batches):
            yield self._batch(bucket)
