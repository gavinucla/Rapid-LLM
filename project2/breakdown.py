"""Break down where the time and memory go for one prefill layer (slide 7).

RAPID-LLM's normal output only gives totals, so this calls its functions directly:
the per-operation time of one global vs one local layer, and the parts of peak memory.
Gemma 3 4B, batch 16, prompt 8192, A100. Writes output/project2/breakdown/breakdown.json.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import config  # noqa: E402
from inference_timing import TimeCalculationLLMInference  # noqa: E402
from memory_estimation import MemoryEstimator  # noqa: E402

BATCH, PROMPT = 16, 8192
GIB = 2 ** 30

hw = config.parse_config(str(REPO / "configs/hardware-config/course_project/a100_80GB_1gpu.yaml"), "hardware")
model = config.parse_config(str(REPO / "configs/model-config/course_project/gemma_3_4b_inf.yaml"), "LLM")
model.model_config.global_batch_size = BATCH
tc = TimeCalculationLLMInference(hw, model, "LLM")

# Time per operation for one global layer and one local layer (window = 1024)
args = (BATCH, tc.vocab_size, tc.hidden_dim, PROMPT, tc.num_heads, tc.kv_heads,
        tc.intermediate_size, hw.tech_config.core.num_bundles)
glob, glob_total = tc.compute_all_gemm_and_node_times(*args, use_moe_override=False)
loc, loc_total = tc.compute_all_gemm_and_node_times(*args, use_moe_override=False, attention_window=1024)


def ms(timings, ops):
    return round(sum(timings[o].total_forward_time() for o in ops if o in timings) * 1000, 1)


groups = {
    "attention": ["attention"],
    "ffn": ["ffn1", "ffn2", "gelu"],
    "projections_and_norms": ["qkv_proj", "output_proj", "layernorm1", "layernorm2"],
}
prefill = {g: {"global_ms": ms(glob, ops), "local_ms": ms(loc, ops)} for g, ops in groups.items()}
prefill["layer_total"] = {"global_ms": round(glob_total["transformer_time_f"] * 1000, 1),
                          "local_ms": round(loc_total["transformer_time_f"] * 1000, 1)}
prefill["output_head_s"] = round(glob_total["linear_softmax_f"], 2)

# Peak memory parts. Weights, embedding and logits are the same with or without the window.
mem = MemoryEstimator(tc).build_memory_data(mode="inference", batch_size=BATCH, seq_len=PROMPT,
                                            kv_cache_tokens=PROMPT)
weights = tc.num_layers * mem["weight_mem_per_layer"] + sum(mem["extra_static_bytes_per_device"].values())
logits = max(v for k, v in mem["transient_bytes_by_kind"].items() if "SOFTMAX" in str(k))
memory = {
    "weights_and_embedding_gib": round(weights / GIB, 2),
    "kv_cache_dense_gib": round(tc.num_layers * mem["kv_cache_bytes_per_layer"] / GIB, 2),
    "kv_cache_window_gib": round(sum(mem["kv_cache_bytes_by_layer"]) / GIB, 2),
    "output_scores_gib": round(logits / GIB, 2),
}

out = REPO / "output/project2/breakdown"
out.mkdir(parents=True, exist_ok=True)
(out / "breakdown.json").write_text(json.dumps({"prefill_one_layer": prefill, "peak_memory": memory}, indent=2))
print(json.dumps({"prefill_one_layer": prefill, "peak_memory": memory}, indent=2))
