from __future__ import annotations

from dataclasses import dataclass

import torch

from gpt2_trace_viewer.domain.trace_step import TraceStep


@dataclass(slots=True)
class AttentionRecord:
    block: int
    head: int
    matrix: torch.Tensor


@dataclass(slots=True)
class TopToken:
    rank: int
    token_id: int
    token: str
    probability: float


@dataclass(slots=True)
class TraceResult:
    prompt: str
    result_text: str
    tokens: list[str]
    steps: list[TraceStep]
    attention_records: list[AttentionRecord]
    top_tokens: list[TopToken]
