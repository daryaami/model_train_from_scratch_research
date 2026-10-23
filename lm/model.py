"""Маленькая decoder-only модель в духе Llama.

Сеть по уже увиденным токенам предсказывает следующий. Словарь и ширина
приходят из конфига, готовых весов нет: обучение с нуля.

Как это связано с остальным кодом:
- data.py подаёт окно токенов и цели, сдвинутые на один токен вправо;
- schedule.py к архитектуре не подключён, он только меняет шаг обучения;
- train.py считает cross-entropy и обновляет веса;
- generate.py берёт logits последней позиции и выбирает следующий id.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

INIT_STD = 0.02


@dataclass
class ModelConfig:
    """Числа архитектуры. Промежуточная ширина SwiGLU считается."""

    vocab_size: int = 16_000
    hidden_size: int = 256
    n_layers: int = 8
    n_heads: int = 4
    context: int = 512
    rope_theta: float = 10_000.0
    rms_eps: float = 1e-5
    swiglu_multiple_of: int = 256

    def __post_init__(self) -> None:
        if self.hidden_size % self.n_heads != 0:
            raise ValueError("hidden_size должен делиться на n_heads без остатка.")
        self.head_dim = self.hidden_size // self.n_heads
        if self.head_dim % 2 != 0:
            raise ValueError("Для RoPE размер головы должен быть чётным.")
        self.intermediate_size = swiglu_width(self.hidden_size, self.swiglu_multiple_of)

    @classmethod
    def from_dict(cls, raw: dict) -> "ModelConfig":
        return cls(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=int(raw["hidden_size"]),
            n_layers=int(raw["n_layers"]),
            n_heads=int(raw["n_heads"]),
            context=int(raw["context"]),
            rope_theta=float(raw["rope_theta"]),
            rms_eps=float(raw["rms_eps"]),
            swiglu_multiple_of=int(raw["swiglu_multiple_of"]),
        )


def swiglu_width(hidden_size: int, multiple_of: int) -> int:
    """Ширина внутреннего слоя SwiGLU, как в Llama.

    Обычный FFN был бы 4 * hidden. У SwiGLU три матрицы вместо двух, поэтому
    внутреннюю ширину уменьшают до 8/3 * hidden и округляют вверх до кратного
    multiple_of, чтобы умножения матриц легли на блоки GPU.
    """
    raw = int(8 * hidden_size / 3)
    if raw % multiple_of == 0:
        return raw
    return raw + multiple_of - (raw % multiple_of)


def count_parameters(model: nn.Module) -> int:
    """Сколько обучаемых чисел в весах. Общая матрица эмбеддинга считается один раз."""
    seen: set[int] = set()
    total = 0
    for param in model.parameters():
        pointer = param.data_ptr()
        if pointer in seen or not param.requires_grad:
            continue
        seen.add(pointer)
        total += param.numel()
    return total


class RMSNorm(nn.Module):
    """Нормализация без вычитания среднего: только масштаб по корню из среднего квадрата.

    В отличие от LayerNorm здесь нет сдвига (bias). Обучается один множитель на канал.
    """

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x.float() * torch.rsqrt(variance + self.eps)
        return normalized.to(dtype=x.dtype) * self.weight


def build_rope_cache(head_dim: int, context: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Таблицы cos и sin для каждой позиции окна.

    Частоты убывают по парам координат головы. Каждую частоту кладём дважды:
    rotate_half крутит первую половину вектора со второй, и обе половины
    должны видеть одну и ту же частоту.
    """
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    inv_freq = 1.0 / (theta ** exponents)
    positions = torch.arange(context, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    freqs = torch.cat((freqs, freqs), dim=-1)
    return freqs.cos(), freqs.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Поворот пары координат: (a, b) переходит в (-b, a)."""
    first, second = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Повернуть query и key на угол, который зависит от позиции токена.

    Value не крутим: позиция кодируется тем, как запрос смотрит на ключ.
    cos и sin имеют форму (T, head_dim) и растягиваются на батч и головы.
    """
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class Attention(nn.Module):
    """Причинное внимание: токен видит себя и то, что левее, но не будущее."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        # is_causal закрывает будущие позиции. Кэш ключей здесь не ведём:
        # и обучение, и короткая генерация передают квадратное окно.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    """Полносвязный блок с воротами: down(silu(gate(x)) * up(x)).

    gate решает, какие каналы пропустить, up несёт сами значения, down
    возвращает ширину модели. Смещений нет, как в Llama.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.intermediate_size
        self.gate = nn.Linear(config.hidden_size, hidden, bias=False)
        self.up = nn.Linear(config.hidden_size, hidden, bias=False)
        self.down = nn.Linear(hidden, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """Один слой: норма, внимание, остаток, норма, SwiGLU, остаток.

    Остаток (x + f(x)) даёт градиенту прямую дорогу через слой, поэтому
    восемь таких блоков можно учить с нуля.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_eps)
        self.attn = Attention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, config.rms_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class LanguageModel(nn.Module):
    """Входные id превращаются в векторы, восемь блоков их перемешивают, logits — это следующий токен.

    Матрица эмбеддинга и матрица logits — одна и та же. Словарь 16 000 при ширине
    256 стоит около 4.1M параметров; вторая копия была бы почти половиной модели.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        cos, sin = build_rope_cache(config.head_dim, config.context, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_module)
        self._scale_residual_projections()

    def _init_module(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)

    def _scale_residual_projections(self) -> None:
        # Выходы внимания и SwiGLU складываются с остатком на каждом слое.
        # Чем больше слоёв, тем сильнее эта сумма раздувается, поэтому
        # стартовые веса этих двух проекций уменьшаем в sqrt(2 * n_layers).
        scale = math.sqrt(2 * self.config.n_layers)
        for block in self.blocks:
            block.attn.o_proj.weight.data.div_(scale)
            block.mlp.down.weight.data.div_(scale)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq = idx.shape
        if seq > self.config.context:
            raise ValueError(f"Окно {seq} длиннее контекста {self.config.context}.")
        x = self.embed(idx)
        cos = self.rope_cos[:seq]
        sin = self.rope_sin[:seq]
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            # Паддинга в потоке нет: id 0 — это EOS, его тоже нужно уметь предсказывать.
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
            )
        return logits, loss


def load_weights(model: LanguageModel, path, device: torch.device) -> dict:
    """Подставить сохранённые веса. Оптимизатор здесь не восстанавливается."""
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    return payload
