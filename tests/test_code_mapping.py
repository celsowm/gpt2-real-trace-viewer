import pytest

from gpt2_trace_viewer.ui.widgets.code_tab import (
    CODE_LINES,
    _STEP_TO_LINE,
    STEP_TO_PATTERN,
    _build_pattern_map,
    find_line,
)

ALL_REAL_KINDS = [
    "Token IDs",
    "Position IDs",
    "Token Embedding",
    "Position Embedding",
    "Embedding Sum",
    "Residual Input",
    "LayerNorm",
    "Linear QKV fused",
    "Q split",
    "K split",
    "V split",
    "Q heads",
    "K heads",
    "V heads",
    "QK scores",
    "Causal Mask",
    "Attention Softmax",
    "Attn Head",
    "Merge Heads",
    "Attention Output Projection",
    "Residual Add",
    "MLP Expand",
    "GELU",
    "MLP Project",
    "Final LayerNorm",
    "Last Token State",
    "Vocabulary Projection",
    "Vocabulary Softmax",
]

STEPS_EXEMPT_FROM_MAPPING = {"Token IDs", "Position IDs", "Attn Head"}


def test_all_patterns_found_in_code_source():
    rebuilt = _build_pattern_map()
    missing = [k for k in STEP_TO_PATTERN if k not in rebuilt]
    assert not missing, f"Patterns not found in CODE_SOURCE: {missing}"


def test_no_dead_patterns():
    rebuilt = _build_pattern_map()
    for key, line_idx in rebuilt.items():
        assert 0 <= line_idx < len(CODE_LINES), (
            f"Key '{key}' -> line {line_idx}, but CODE_LINES has {len(CODE_LINES)} lines"
        )
        assert CODE_LINES[line_idx].strip(), (
            f"Key '{key}' -> line {line_idx}, which is empty"
        )


def test_coverage_of_real_step_kinds():
    missing = []
    for kind in ALL_REAL_KINDS:
        if kind in STEPS_EXEMPT_FROM_MAPPING:
            continue
        found = find_line("", kind)
        if found < 0:
            missing.append(kind)
    assert not missing, f"Real step kinds without mapping: {missing}"


def test_find_line_returns_valid():
    assert find_line("anything", "Embedding Sum") >= 0
    assert find_line("anything", "MLP Expand") >= 0
    assert find_line("anything", "GELU") >= 0
    assert find_line("anything", "Vocabulary Projection") >= 0
