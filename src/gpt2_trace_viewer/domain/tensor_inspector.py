from __future__ import annotations

from typing import Any

import torch


def tensor_stats(tensor: torch.Tensor) -> dict[str, str]:
    """Return display-safe statistics for any tensor."""
    t = tensor.detach().float().cpu()
    flat = t.reshape(-1)

    if flat.numel() == 0:
        return {
            "shape": str(tuple(t.shape)),
            "min": "n/a",
            "max": "n/a",
            "mean": "n/a",
            "std": "n/a",
            "l2": "n/a",
            "abs_mean": "n/a",
        }

    return {
        "shape": str(tuple(t.shape)),
        "min": f"{flat.min().item():.6f}",
        "max": f"{flat.max().item():.6f}",
        "mean": f"{flat.mean().item():.6f}",
        "std": f"{flat.std(unbiased=False).item():.6f}",
        "l2": f"{torch.linalg.vector_norm(flat).item():.6f}",
        "abs_mean": f"{flat.abs().mean().item():.6f}",
    }


def top_abs_values(tensor: torch.Tensor, k: int = 20) -> list[dict[str, str]]:
    """Return the largest absolute values with original tensor indices."""
    detached = tensor.detach().float().cpu()
    flat = detached.reshape(-1)

    if flat.numel() == 0:
        return []

    k = min(k, flat.numel())
    abs_values, flat_indices = torch.topk(flat.abs(), k)

    rows: list[dict[str, str]] = []
    original_shape = tuple(detached.shape)

    for abs_value, flat_index_tensor in zip(abs_values, flat_indices):
        flat_index = int(flat_index_tensor.item())
        value = flat[flat_index].item()
        multi_index = _flat_to_multi_index(flat_index, original_shape)

        rows.append(
            {
                "index": str(multi_index),
                "value": f"{value:.6f}",
                "abs": f"{abs_value.item():.6f}",
            }
        )

    return rows


def _flat_to_multi_index(flat_index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return tuple()

    values: list[int] = []
    remainder = flat_index

    for size in reversed(shape):
        values.append(remainder % size)
        remainder //= size

    return tuple(reversed(values))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default
