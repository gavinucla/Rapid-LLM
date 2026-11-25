"""Make the plots for the slides from output/project2/*/summary.json.

Charts (from the runs):   context.png (slide 10), window_sweep.png (slide 11), h100.png (slide 13)
Diagrams (no data):       layers.png (slide 3), tiles.png (slide 4)

Everything goes to output/project2/plots/.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "output" / "project2"
PLOTS = RESULTS / "plots"

BLUE, LIGHT_BLUE, ORANGE, GRAY = "#2774AE", "#DAE6F4", "#E8743B", "#57585B"
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#E6E6E6", "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})
CONTEXTS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]


def load(study):
    rows = json.loads((RESULTS / study / "summary.json").read_text())
    return {r["name"]: r for r in rows}


def label(n):
    return f"{n // 1024}K" if n >= 1024 else str(n)


def speedups(runs, metric):
    """Dense / windowed for each prompt length in a context sweep."""
    return [float(runs[f"dense_p{p}"][metric]) / float(runs[f"window_p{p}"][metric]) for p in CONTEXTS]


def log_x(ax, ticks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks, [label(t) for t in ticks])


# ---- charts -------------------------------------------------------------------------------

def context_plot():
    runs = load("context")
    prefill, tpot = speedups(runs, "prefill_s"), speedups(runs, "time_per_token_ms")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(CONTEXTS, prefill, "-o", color=BLUE, label="Prefill speedup")
    ax.plot(CONTEXTS, tpot, "-s", color=ORANGE, label="Time-per-token speedup")
    for p, v in zip(CONTEXTS, prefill):
        if p >= 8192:
            ax.annotate(f"{v:.2f}x", (p, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9, color=BLUE)
    for p, v in zip(CONTEXTS, tpot):
        if p >= 32768:
            ax.annotate(f"{v:.2f}x", (p, v), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=9, color=ORANGE)
    ax.axvline(1024, color=GRAY, ls=":", lw=1)
    ax.text(1100, 1.45, "w = 1,024", color=GRAY, fontsize=9)
    ax.axhline(1, color="#BBBBBB", lw=1)
    log_x(ax, CONTEXTS)
    ax.set(xlabel="Prompt length (tokens)", ylabel="Windowed vs dense (x)", ylim=(0.8, 4.2))
    ax.legend(loc="upper left")
    fig.savefig(PLOTS / "context.png")
    plt.close(fig)


def window_sweep_plot():
    runs = load("sweeps")
    windows = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    tpot = [float(runs[f"window_{w}"]["time_per_token_ms"]) for w in windows]
    prefill = [float(runs[f"window_{w}"]["prefill_s"]) for w in windows]
    dense_tpot, dense_prefill = float(runs["dense"]["time_per_token_ms"]), float(runs["dense"]["prefill_s"])

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(windows, tpot, "-o", color=BLUE, label="Time per token (ms)")
    ax.axhline(dense_tpot, color=BLUE, ls="--", lw=1)
    ax.text(3600, dense_tpot + 0.15, f"dense {dense_tpot:.1f} ms", color=BLUE, fontsize=9, ha="right")
    log_x(ax, windows)
    ax.set(xlabel="Window size w (tokens)", ylim=(9, 19))
    ax.set_ylabel("Time per token (ms)", color=BLUE)

    ax2 = ax.twinx()
    ax2.plot(windows, prefill, "-s", color=ORANGE, label="Prefill (s)")
    ax2.axhline(dense_prefill, color=ORANGE, ls="--", lw=1)
    ax2.text(3600, dense_prefill - 0.09, f"dense {dense_prefill:.2f} s", color=ORANGE, fontsize=9, ha="right")
    ax2.set_ylabel("Prefill (s)", color=ORANGE)
    ax2.set_ylim(4.0, 6.0)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)

    ax.axvline(1024, color=GRAY, ls=":", lw=1)
    ax.text(1100, 9.3, "Gemma 3", color=GRAY, fontsize=9)
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax.legend(lines, [l.get_label() for l in lines], loc="center left", bbox_to_anchor=(0, 0.62))
    fig.savefig(PLOTS / "window_sweep.png")
    plt.close(fig)


def h100_plot():
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for study, color, gpu in (("context", BLUE, "A100"), ("h100", ORANGE, "H100")):
        runs = load(study)
        prefill, tpot = speedups(runs, "prefill_s"), speedups(runs, "time_per_token_ms")
        ax.plot(CONTEXTS, prefill, "-o", color=color, label=f"{gpu} prefill")
        ax.plot(CONTEXTS, tpot, "--s", color=color, label=f"{gpu} time per token", alpha=0.85)
        ax.annotate(f"{prefill[-1]:.2f}x", (CONTEXTS[-1], prefill[-1]), textcoords="offset points",
                    xytext=(-4, 6), ha="right", fontsize=9, color=color)
    ax.axhline(1, color="#BBBBBB", lw=1)
    log_x(ax, CONTEXTS)
    ax.set(xlabel="Prompt length (tokens)", ylabel="Windowed vs dense (x)")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(PLOTS / "h100.png")
    plt.close(fig)


# ---- diagrams -----------------------------------------------------------------------------

def layers_diagram():
    """Gemma 3's 34 layers: every 6th one is global."""
    fig, ax = plt.subplots(figsize=(10, 1.5))
    for i in range(34):
        is_global = (i + 1) % 6 == 0
        ax.add_patch(plt.Rectangle((i, 0), 0.9, 1, color=BLUE if is_global else LIGHT_BLUE))
        if is_global:
            ax.text(i + 0.45, 0.5, str(i), ha="center", va="center", color="white", fontsize=9, weight="bold")
    ax.set(xlim=(-0.2, 34), ylim=(0, 1))
    ax.axis("off")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=LIGHT_BLUE), plt.Rectangle((0, 0), 1, 1, color=BLUE)],
              labels=["Local (last 1,024 tokens): 29 layers", "Global (all tokens): 5 layers"],
              loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=10)
    fig.savefig(PLOTS / "layers.png")
    plt.close(fig)


def tiles_diagram():
    """FlashAttention tiles, scaled down to 10x10 with a 3-tile window."""
    n, window = 10, 3
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, title, w in ((axes[0], "Global layer: triangle", None), (axes[1], "Local layer: band", window)):
        count = 0
        for row in range(n):
            for col in range(n):
                if col > row:
                    color = "white"  # future tokens, never computed
                elif w is not None and col < row - w:
                    color = "#EEEEEE"  # older than the window, skipped
                else:
                    color = BLUE
                    count += 1
                ax.add_patch(plt.Rectangle((col, n - 1 - row), 0.92, 0.92, color=color))
        ax.set(xlim=(-0.1, n), ylim=(-0.1, n), aspect="equal")
        ax.axis("off")
        ax.set_title(f"{title}\n{count} of 100 tiles computed", color=GRAY, fontsize=11)
    fig.text(0.5, 0.0, "rows = query tiles, columns = key tiles (scaled down: 10x10 tiles, window = 3 tiles)",
             ha="center", fontsize=9, color="#808080")
    fig.savefig(PLOTS / "tiles.png")
    plt.close(fig)


if __name__ == "__main__":
    PLOTS.mkdir(parents=True, exist_ok=True)
    for make in (context_plot, window_sweep_plot, h100_plot, layers_diagram, tiles_diagram):
        make()
    print(f"plots written to {PLOTS}")
