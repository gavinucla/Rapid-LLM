"""Run the Project 2 case studies (sliding-window attention on Gemma 3 4B).

Each run writes a model YAML, calls run_perf.py in its own folder, and pulls the numbers out
of the log. Every study ends up as output/project2/<study>/summary.csv (and .json).

    python project2/run_studies.py            # everything
    python project2/run_studies.py context    # just one study
"""

import csv
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "output" / "project2"
GEMMA = REPO / "configs/model-config/course_project/gemma_3_4b_inf.yaml"
MISTRAL = REPO / "project2/configs/mistral_7b_inf.yaml"
GPUS = {  # name: (hardware config, memory in GiB)
    "a100": (REPO / "configs/hardware-config/course_project/a100_80GB_1gpu.yaml", 80),
    "h100": (REPO / "configs/hardware-config/course_project/H100_SXM5_80GB.yaml", 80),
    "t4": (REPO / "project2/configs/t4_16GB_1gpu.yaml", 16),
}

# Attention designs from the papers. interval = every Nth layer is global
# (an interval bigger than the layer count makes every layer local).
DESIGNS = {
    "dense": None,
    "gemma3": {"window_size": 1024, "local_global_interval": 6},
    "gemma2": {"window_size": 4096, "local_global_interval": 2},
    "gptoss": {"window_size": 128, "local_global_interval": 2},
    "mistral": {"window_size": 4096, "local_global_interval": 1000},
}
KEEP = "keep"  # use the window already in the model YAML


def run(job):
    """Run one config and return its metrics."""
    name, model, gpu, batch, prompt, output, window, extra = job
    folder = OUT / "runs" / name
    folder.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(model.read_text())
    mp = cfg["model_param"]
    mp["global_batch_size"] = batch
    mp["seq_len"] = prompt + output  # RAPID-LLM's seq_len is prompt + output
    mp["decode_len"] = output
    if window is None:
        mp["attention"].pop("window", None)
    elif window != KEEP:
        mp["attention"]["window"] = {"last_layer_global": False, **window}
    (folder / "model.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    hw, capacity = GPUS[gpu]
    proc = subprocess.run(
        [sys.executable, str(REPO / "run_perf.py"), "--hardware_config", str(hw),
         "--model_config", str(folder / "model.yaml")],
        cwd=folder, capture_output=True, text=True,
    )
    log = proc.stdout + proc.stderr
    (folder / "console.log").write_text(log)
    mem_file = folder / "output/LLM/memory-summary/memory_capacity_comparison.txt"
    mem = mem_file.read_text() if mem_file.exists() else ""

    def grab(pattern, text=log):
        m = re.search(pattern, text)
        return m.group(1) if m else None

    decode = grab(r"\[decode\] time: ([\d.]+)s")
    peak = grab(r"Max peak memory usage \(per gpu\): ([\d.]+)", mem)
    row = {
        "name": name, "gpu": gpu, "batch": batch, "prompt": prompt, "output": output, **extra,
        "prefill_s": grab(r"\[prefill\] time: ([\d.]+)s"),
        "time_per_token_ms": round(float(decode) / output * 1000, 3) if decode else None,
        "kv_cache": grab(r"prefill_store=([\d.]+ \w+)"),
        "kv_read_decode": grab(r"decode_fetch=([\d.]+ \w+)"),
        "peak_gib": peak,
        "fits": peak is not None and float(peak) <= capacity,
        "ok": proc.returncode == 0,
    }
    print(f"  {name}: prefill {row['prefill_s']}s, {row['time_per_token_ms']} ms/token, peak {peak} GiB", flush=True)
    return row


def job(name, batch, prompt, output=128, window=KEEP, gpu="a100", model=GEMMA, **extra):
    return (name, model, gpu, batch, prompt, output, window, extra)


# ---- the studies -------------------------------------------------------------------------

def required_eval():
    """Batch 16, prompt 8K, output 128 and 256, dense vs windowed (slide 6)."""
    return [job(f"{a}_out{o}", 16, 8192, o, None if a == "dense" else KEEP, attn=a)
            for o in (128, 256) for a in ("dense", "window")]


def window_check():
    """A window covering the whole sequence should match dense exactly."""
    return [job("dense", 16, 8192, window=None),
            job("w16384", 16, 8192, window={"window_size": 16384, "local_global_interval": 6})]


CONTEXTS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]


def context(gpu="a100"):
    """Batch 1, prompt 512 to 128K (slides 10 and 13)."""
    return [job(f"{a}_p{p}", 1, p, window=None if a == "dense" else KEEP, gpu=gpu, attn=a)
            for p in CONTEXTS for a in ("dense", "window")]


def designs():
    """Each paper's ratio and window applied to the Gemma 3 4B shape (slide 8)."""
    jobs = []
    for d, w in DESIGNS.items():
        jobs.append(job(f"{d}_b16_p8K", 16, 8192, window=w, design=d))
        jobs.append(job(f"{d}_b1_p32K", 1, 32768, window=w, design=d))
    return jobs


def sweeps():
    """Window size at 5:1, and local:global ratio at w = 1024 (slides 11 and 12)."""
    jobs = [job("dense", 16, 8192, window=None, sweep="dense")]
    for w in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
        jobs.append(job(f"window_{w}", 16, 8192, sweep="window", w=w,
                        window={"window_size": w, "local_global_interval": 6}))
    for ratio in (1, 3, 5, 7):
        jobs.append(job(f"ratio_{ratio}to1", 16, 8192, sweep="ratio", ratio=f"{ratio}:1",
                        window={"window_size": 1024, "local_global_interval": ratio + 1}))
    return jobs


def mistral():
    """Mistral 7B, all layers windowed at 4096, vs the same model dense (slide 9)."""
    return [job(f"{a}_p{p}", 1, p, model=MISTRAL, window=None if a == "dense" else KEEP, attn=a)
            for p in (16384, 32768) for a in ("dense", "swa")]


def h100():
    """Required eval and context sweep on the H100 (slide 13)."""
    return [job(f"{a}_out128", 16, 8192, gpu="h100", window=None if a == "dense" else KEEP, attn=a)
            for a in ("dense", "window")] + context("h100")


def t4():
    """Batch 1 on a 16 GB T4 (slide 14)."""
    return [job(f"{a}_p{p}", 1, p, gpu="t4", window=None if a == "dense" else KEEP, attn=a)
            for p in (8192, 16384) for a in ("dense", "window")]


STUDIES = {
    "required_eval": required_eval, "window_check": window_check, "context": context,
    "designs": designs, "sweeps": sweeps, "mistral": mistral, "h100": h100, "t4": t4,
}


def save(study, rows):
    folder = OUT / study
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(json.dumps(rows, indent=2))
    with open(folder / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def run_study(study, jobs):
    # prefix run folders with the study name so studies don't overwrite each other
    jobs = [(f"{study}/{j[0]}",) + j[1:] for j in jobs]
    print(f"{study}: {len(jobs)} runs", flush=True)
    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(run, jobs))
    for r in rows:
        r["name"] = r["name"].split("/", 1)[1]
    save(study, rows)
    return rows


def largest_batch():
    """Biggest batch that fits in 80 GB at prompt 8K, output 256: double, then bisect (slide 6)."""
    results = []
    for attn in ("dense", "window"):
        window = None if attn == "dense" else KEEP

        def fits(b):
            return run(job(f"largest_batch/{attn}_b{b}", b, 8192, 256, window))["fits"]

        lo, hi = 0, 1
        while fits(hi):
            lo, hi = hi, hi * 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if fits(mid) else (lo, mid)
        print(f"largest batch, {attn}: {lo}", flush=True)
        results.append({"attn": attn, "prompt": 8192, "output": 256, "largest_batch": lo})
    save("largest_batch", results)


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(STUDIES) + ["largest_batch"]
    for study in wanted:
        if study == "largest_batch":
            largest_batch()
        else:
            run_study(study, STUDIES[study]())
