# RAPID-LLM (course branch)

RAPID-LLM estimates how long a large language model takes to run on given hardware, and how much
GPU memory it needs. It does not run the model. It turns the model shape (layers, hidden size,
heads, experts) into matrix multiplications and other operations, prices each one from the GPU's
compute rate and memory bandwidth, adds communication time when several GPUs work together, and
adds everything up.

This course branch runs on the analytical backend. This file covers setup, running a config and
how RAPID-LLM prices a decode step. For the course projects, read
[COURSE_README.md](COURSE_README.md): the helpers we added and where to start in the code.

## Project 2: Sliding-window attention

This fork adds Gemma 3-style sliding-window attention to RAPID-LLM. Some layers are "local" and only
look at the last `w` tokens; the rest are "global" and see everything. Gemma 3 4B uses 5 local
layers for every global one, with `w = 1024`.

RAPID-LLM used to refuse a model with a `window:` block. Now it prices local layers properly:

- **Decode:** a local layer reads at most `w` tokens of KV cache.
- **Prefill:** a local layer only computes the band of FlashAttention tiles near the diagonal,
  instead of the full triangle.
- **Memory:** a local layer only keeps `w` tokens of KV cache.

Global layers are unchanged, and models without a `window:` block give exactly the same numbers as
before.

### Setup

Python 3.12, from the repo root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt pytest -r project2/requirements.txt
pytest -q          # 43 passed, 2 skipped
```

`project2/requirements.txt` only adds matplotlib, for the plots.

### Reproduce everything

```bash
python reproduce.py
```

This runs all the case studies, the slide 7 breakdown, and the plots. It's about 80 RAPID-LLM runs
(8 in parallel) and took about 25 minutes on a 24-core machine. Everything lands in
`output/project2/`.

You can also run one study at a time, e.g. `python project2/run_studies.py context`. The study
names are `required_eval`, `window_check`, `context`, `designs`, `sweeps`, `mistral`, `h100`, `t4`
and `largest_batch`.

To run the Gemma 3 config by hand:

```bash
python run_perf.py --hardware_config configs/hardware-config/course_project/a100_80GB_1gpu.yaml \
                   --model_config configs/model-config/course_project/gemma_3_4b_inf.yaml
```

### Where each slide's numbers come from

Each study folder has a `summary.csv` (and the same thing as `summary.json`). The raw RAPID-LLM log
and model YAML for every run are in `output/project2/runs/<study>/<run>/`.

| Slide | What it shows | Output |
|---|---|---|
| 3 | Local vs global layers | `plots/layers.png` |
| 4 | Triangle vs band of tiles; window ≥ sequence matches dense | `plots/tiles.png`, `window_check/summary.csv` |
| 6 | Required eval (batch 16, prompt 8K, output 128 and 256), largest batch | `required_eval/summary.csv`, `largest_batch/summary.csv` |
| 7 | Time per op in one prefill layer, and what makes up peak memory | `breakdown/breakdown.json` |
| 8 | Gemma 2, gpt-oss and Mistral attention designs on the Gemma 3 4B shape | `designs/summary.csv` |
| 9 | Mistral 7B, checked against the paper | `mistral/summary.csv` |
| 10 | Speedup vs prompt length | `plots/context.png`, `context/summary.csv` |
| 11 | Window size sweep | `plots/window_sweep.png`, `sweeps/summary.csv` (rows `window_*`) |
| 12 | Local:global ratio sweep | `sweeps/summary.csv` (rows `ratio_*`) |
| 13 | A100 vs H100 | `plots/h100.png`, `h100/summary.csv` |
| 14 | T4 (16 GB): dense doesn't fit at 16K, windowed does | `t4/summary.csv` |

All paths are under `output/project2/`. A few numbers on the slides are from the papers, not
RAPID-LLM: the MMLU scores on slide 8 and the "~2x" and "8x" claims on slide 9.

The columns in `summary.csv`: `prefill_s` is prefill time, `time_per_token_ms` is decode time
divided by output length, `kv_cache` is KV kept after prefill, `kv_read_decode` is KV read during
decode, `peak_gib` is peak GPU memory, and `fits` says whether that's under the GPU's memory (80 GB,
or 16 GB for the T4).

### What changed in the code

Six RAPID-LLM files changed (about 140 lines added), plus one new test file. Every change only
kicks in when the model YAML has a `window:` block, so dense models give the same numbers as before.

- **`llm_util.py`**: three small helpers that hold the local/global rule in one place, so
  decode, prefill and memory can't disagree about which layers are local.
  `layer_is_global` says whether layer `i` is global (every 6th layer for Gemma 3),
  `layer_block_kinds` returns `"global"`/`"local"` for every layer, and `attended_tokens` gives
  how many tokens a layer sees: the whole context if global, `min(context, w)` if local.
- **`base_timing.py`**: copies the parsed window onto the timing object as `attention_window`
  (2 lines), the same way the other attention settings are copied.
- **`config.py`**: removes the check that refused models with a `window:` block.
- **`memory_estimation.py`**: fills in course helper H3, a list with one KV cache size per
  layer. A local layer's size is the full-context size scaled by `min(context, w) / context`.
- **`inference_timing.py`**:
  - *Decode:* prices one extra "local" layer at a context of `w` and uses course helper H4 to
    give each layer its own time (5 global layers at the full context, 29 local ones at `w`).
  - *Prefill:* same idea, but the local layer is priced with the windowed FlashAttention below.
  - *`[kv-cache]` line:* the printed KV stored and KV read now count local layers at `w`.
- **`train_timing.py`**: `flash_attention_kernel_forward` takes an optional `window`. For each
  row of query tiles it already found the last key tile it needs (the diagonal); now it also
  finds the first one, `min_c`, and skips the tiles before it. That turns the triangle into a
  band. `compute_all_gemm_and_node_times` passes the window through.
- **`tests/test_sliding_window.py`** (new): checks the layer rule (Gemma 3's global layers are
  5, 11, 17, 23, 29) and that the KV cache comes out at 4.3 GiB instead of 17.0 GiB.

Project files:

| File | What it does |
|---|---|
| `reproduce.py` | Runs everything below |
| `project2/run_studies.py` | All the case studies |
| `project2/breakdown.py` | Slide 7's per-op times and memory parts (calls RAPID-LLM's functions directly) |
| `project2/make_plots.py` | The three charts and two diagrams |
| `project2/configs/` | Mistral 7B model config and a T4 hardware config (the course configs aren't touched) |

### Caveats

- RAPID-LLM keeps the output scores (logits) for every prompt token during prefill. With Gemma's
  262k vocabulary that's 64 GiB at batch 16, prompt 8K, and it's most of the prefill peak. That's
  why the largest batch only goes from 14 to 17 even though the KV cache shrinks about 4x.
- A tile that's only partly inside the window is counted in full, same as the existing code does
  for tiles on the diagonal. So prefill stops improving below `w = 128` (the tile size).
- The T4 config uses datasheet numbers and isn't tuned against real measurements, so treat its
  absolute times as rough. Dense vs windowed on the same GPU is still a fair comparison.
- RAPID-LLM doesn't measure quality. Quality claims on the slides are from the papers.

---

*Everything below is the course's original README. For this project, the setup above already
covers it.*

## Setup (Python 3.12)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt pytest
pytest -q                           # should pass
```

## Running one config

```bash
python run_perf.py --hardware_config configs/hardware-config/course_project/a100_80GB_1gpu.yaml \
                   --model_config configs/model-config/course_project/Llama3.1-8B_inf.yaml
```

Results go to `output/LLM/`: `LLM_inference_results.txt` repeats the summary, and
`memory-summary/memory_capacity_comparison.txt` gives the prefill, decode and maximum peak memory
per GPU. The lines that matter on screen:

```
[decode] sample step 0: seq_len=1025, time=0.0121s
[decode] sample step 1023: seq_len=2048, time=0.0122s
[prefill] time: 0.0661s, [decode] time: 12.4618s, [total] time: 12.5279s
[kv-cache] prefill_store=0.1 GiB, decode_store=0.1 GiB, decode_fetch=192.0 GiB
LLM time to first token: 0.08s
Aggregate decode throughput tok/s (batch=1, replica_count=1): start=82.40, mid(token 512)=82.17, end=81.94
```

- `sample step`: one decode step (one new token for every sequence) at that context length.
- `prefill` processes the prompt once; `decode` generates all output tokens; time to first token
  is prefill plus the first decode step.
- `kv-cache`: key/value bytes written in prefill and decode and read in decode. `tok/s`: tokens
  per second over the whole batch at the start, middle and end of decoding.

## Configs

Put all your settings in the model config. The hardware configs are shared: do not edit them.

`configs/model-config/course_project/` holds the four project models, and
`configs/hardware-config/course_project/` the GPUs used in the course.

| Model config | What it is |
|---|---|
| `course_project/qwen3_30b_a3b_inf.yaml` | Qwen3-30B-A3B, mixture of experts (Project 1) |
| `course_project/gemma_3_4b_inf.yaml` | Gemma 3 4B text model with sliding window (Project 2) |
| `course_project/Llama3.1-8B_inf.yaml` | Llama-3.1-8B, dense (Project 3) |
| `course_project/mimo_7b_inf.yaml` | MiMo-7B base model, dense (Project 4) |
| `Llama3.1-70B_inf.yaml`, `Llama3.1-405B_inf.yaml` | larger dense Llama models |
| `DeepSeekV3_inf.yaml`, `GLM4.7_358B_inf.yaml` | large mixture-of-experts models |
| `minicpm3_4b_inf.yaml` | a small dense model with MLA attention |

| Hardware config | What it is |
|---|---|
| `course_project/a100_80GB_1gpu.yaml` | one A100 80 GB (all project evaluations) |
| `course_project/a100_80GB.yaml` | two A100 80 GB, tensor parallel (tp: 2) |
| `course_project/H100_SXM5_80GB.yaml` | one H100 SXM5 80 GB |
| `waferscale_20v100_80hbm.yaml` | 20 V100-class GPUs in one node, tensor parallel |

The Gemma file is refused until Project 2 is done; delete its `window:` block to run it as a dense
model. Units: in the YAML files `GB` means 2^30 bytes (as a bandwidth, 2^30 bytes per second);
latencies are in seconds.

## Changing the inputs

In the model YAML, under `model_param`: `global_batch_size` is the number of sequences served
together (B), `decode_len` the decode (output) tokens per sequence, and `seq_len` the prefill plus
decode tokens; to change only the output length, change both by the same amount. Under
`inference_param`, `sample_every: -1` (the default) prices only the first and last decode step;
`sample_every: N` also prices every Nth step.

Your own inputs (link bandwidth and latency, window size, block size, number of guesses, and so
on) go in a `project_param:` block in the model YAML; see H1 in [COURSE_README.md](COURSE_README.md).

## How a decode step is priced

1. `TimeCalculationLLMInference.calc_total_inference_time` (inference_timing.py) runs prefill
   and peak memory in `calc_time`, then decode in `calc_decode_time`, which hands over to
   `DecodeGraph.build_decode_graph` (simulate_inference_graph.py).
2. `build_decode_graph` picks the sampled steps (`_generate_sample_points`; by default only the
   first and last step), computes the GEMM shapes for the current context length
   (`llm_util.process_decode_gemm_shapes`, which calls `autoregressive_decoder_gemm`) and calls
   `_execute_decode_step`.
3. `_execute_decode_step` makes a fresh timing object and calls `prepare_decode_graphs`
   (inference_timing.py).
4. `prepare_decode_graphs` calls `_build_decode_transformer_results` for a dense layer and, for
   MoE models, an MoE layer. Each call returns a table of operations (`qkv_proj`, `attention`,
   `ffn1`, ...) and a `node_breakdown` holding the per-layer time `transformer_time_f` and the
   output-head time `linear_softmax_f`.
5. `_prepare_execution_graphs` (train_timing.py) turns these into a graph of layers
   (simulate_train_graph.py), and `LLMExecutionDispatcher.run` (llm_execution.py) returns the
   step time.
6. `_integrate_decode_samples` joins the sampled steps with straight lines (trapezoid rule) to
   get the total decode time.

Peak memory: `calc_time` calls `MemoryEstimator.build_memory_data` (memory_estimation.py) for
weights, KV cache and activations, then `simulate_peak`, which replays the layer graph
(simulate_train_graph.py). Going over the GPU's memory prints a warning; it does not stop the run.
