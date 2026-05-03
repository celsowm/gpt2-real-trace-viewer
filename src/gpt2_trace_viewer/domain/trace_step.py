from __future__ import annotations

from dataclasses import dataclass, field

import torch

from gpt2_trace_viewer.domain.tensor_inspector import tensor_stats


@dataclass(slots=True)
class TraceStep:
    name: str
    kind: str
    tensor: torch.Tensor
    block: int | None = None
    head: int | None = None
    description: str = ""
    extra: dict[str, str] = field(default_factory=dict)
    stats: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        self.tensor = self.tensor.detach().cpu()
        self.stats = tensor_stats(self.tensor)
