import pytest
import torch


def test_weights_loaded_correctly():
    from gpt2_trace_viewer.gpt2_real import GPT2LMHeadModel
    model = GPT2LMHeadModel()
    assert model.lm_head.weight.shape == (50257, 768)
    assert model.transformer.wte.weight.shape == (50257, 768)
    assert len(model.transformer.h) == 12


def test_forward_pass_shape():
    from gpt2_trace_viewer.gpt2_real import GPT2LMHeadModel
    model = GPT2LMHeadModel()
    model.eval()
    input_ids = torch.tensor([[28228, 262, 1368]])
    with torch.no_grad():
        logits = model(input_ids)
    assert logits.shape == (1, 3, 50257), f"Expected (1, 3, 50257), got {logits.shape}"


def test_inference_produces_valid_token():
    from gpt2_trace_viewer.gpt2_real import GPT2LMHeadModel
    model = GPT2LMHeadModel()
    model.eval()
    input_ids = torch.tensor([[28228, 262, 1368, 373, 1026, 4953]])
    with torch.no_grad():
        logits = model(input_ids)
        next_token = logits[0, -1].argmax().item()
    assert isinstance(next_token, int), "next_token should be an int"
    assert 0 <= next_token < 50257, f"next_token {next_token} out of vocab range"


def test_main_execution():
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "gpt2_trace_viewer.gpt2_real"],
        capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, f"gpt2_real.py failed: {result.stderr}"
    assert "Next token:" in result.stdout