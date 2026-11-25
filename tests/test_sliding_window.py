"""Project 2: sliding-window layer rule helpers in llm_util."""

import config
import llm_util


def _window(w=1024, n=6, last_global=False):
    return config.AttentionWindowConfig(
        window_size=w, local_global_interval=n, last_layer_global=last_global
    )


def test_gemma3_global_layers():
    win = _window()
    types = llm_util.layer_block_kinds(34, win)
    assert [i for i, t in enumerate(types) if t == "global"] == [5, 11, 17, 23, 29]
    assert types.count("local") == 29


def test_last_layer_global_adds_one():
    types = llm_util.layer_block_kinds(34, _window(last_global=True))
    assert [i for i, t in enumerate(types) if t == "global"] == [5, 11, 17, 23, 29, 33]


def test_no_window_or_interval_one_is_dense():
    assert llm_util.layer_block_kinds(34, None) == ["global"] * 34
    assert llm_util.layer_block_kinds(34, _window(n=1)) == ["global"] * 34


def test_attended_tokens():
    win = _window(w=1024)
    # local layer: capped at w once context exceeds it
    assert llm_util.attended_tokens(0, 8192, 34, win) == 1024
    assert llm_util.attended_tokens(0, 500, 34, win) == 500
    # global layer: full context
    assert llm_util.attended_tokens(5, 8192, 34, win) == 8192
    # dense model: full context everywhere
    assert llm_util.attended_tokens(0, 8192, 34, None) == 8192


# --- Part 1: per-layer KV memory (helper H3) ---------------------------------

from pathlib import Path

from inference_timing import TimeCalculationLLMInference
from memory_estimation import MemoryEstimator

_ROOT = Path(__file__).resolve().parents[1]
_HW = _ROOT / "configs/hardware-config/course_project/a100_80GB_1gpu.yaml"
_GEMMA = _ROOT / "configs/model-config/course_project/gemma_3_4b_inf.yaml"


def _gemma_kv(drop_window: bool):
    hw = config.parse_config(str(_HW), config_type="hardware")
    model = config.parse_config(str(_GEMMA), config_type="LLM")
    if drop_window:
        model.model_config.attention.window = None
    tc = TimeCalculationLLMInference(hw, model, "LLM")
    return tc, MemoryEstimator(tc).build_memory_data(
        mode="inference", batch_size=16, seq_len=8192, kv_cache_tokens=8192
    )


def test_window_reaches_timing_object():
    tc, _ = _gemma_kv(drop_window=False)
    assert tc.attention_window.window_size == 1024


def test_windowed_kv_by_layer():
    _, data = _gemma_kv(drop_window=False)
    gib = 2 ** 30
    full = data["kv_cache_bytes_per_layer"]
    assert full == 0.5 * gib  # 16 * 4 * 256 * 8192 * 2 B * 2 (K,V)
    by_layer = data["kv_cache_bytes_by_layer"]
    assert len(by_layer) == 34
    assert by_layer[5] == full            # global layer: full context
    assert by_layer[0] == full / 8        # local layer: 1024 of 8192 tokens
    assert sum(by_layer) == 5 * full + 29 * full / 8  # 4.3125 GiB


def test_dense_has_no_kv_by_layer():
    _, data = _gemma_kv(drop_window=True)
    assert data["kv_cache_bytes_by_layer"] is None
