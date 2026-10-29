
import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

START = "corner"   # "center" | "corner": punto di partenza del test da rappresentare

# per ogni partenza: file dei dati e suffisso dei grafici (il test dal centro mantiene i nomi originali)
DATASETS = {
    "center": ("final_50seeds", ""),
    "corner": ("benchmark_macro4_corner_50seeds", "_corner"),
}
DATA_NAME, OUTPUT_SUFFIX = DATASETS[START]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
COVERAGE_PATH = os.path.join(RESULTS_DIR, f"{DATA_NAME}_coverage.jsonl")
RUNS_PATH = os.path.join(RESULTS_DIR, f"{DATA_NAME}.csv")
OUTPUT_BASE = os.path.join(RESULTS_DIR, f"coverage_vs_distance{OUTPUT_SUFFIX}")
FINAL_OUTPUT_BASE = os.path.join(RESULTS_DIR, f"final_coverage_per_run{OUTPUT_SUFFIX}")

BLOCKED_THRESHOLD = 90.0

RESOLUTION = 0.2
TOTAL_CELLS = int(20.0 / RESOLUTION) * int(20.0 / RESOLUTION)

SERIES = {
    "random":     {"label": "Random",                       "color": "#2a78d6"},
    "frontier":   {"label": "Frontier-based",               "color": "#eb6834"},
    "serpentine": {"label": "Serpentina", "color": "#1baf7a"},
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def load_curves():
    curves = {name: [] for name in SERIES}
    with open(COVERAGE_PATH) as f:
        for line in f:
            run = json.loads(line)
            curves[run["strategy"]].append(np.array(run["coverage_history"]) / TOTAL_CELLS * 100)
    return curves


def style_axes(ax, grid_axis):
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)


def mean_curve(runs, length):
    # un run che si e' fermato (completato o bloccato) resta alla sua copertura finale,
    # cosi' la media a ogni distanza e' sempre calcolata su tutti i run
    padded = np.array([np.pad(r, (0, length - len(r)), mode="edge") for r in runs])
    return padded.mean(axis=0)


def spread_labels(values, min_gap):
    # etichette con valori troppo vicini si sovrappongono: le distanzio di almeno min_gap,
    # spostando il gruppo in modo che resti centrato sui valori reali
    order = sorted(values, key=values.get, reverse=True)
    positions = {}
    for i, name in enumerate(order):
        positions[name] = values[name] if i == 0 else min(values[name], positions[order[i - 1]] - min_gap)
    shift = np.mean([values[name] - positions[name] for name in order])
    return {name: positions[name] + shift for name in order}


def main():
    curves = load_curves()
    length = max(len(r) for runs in curves.values() for r in runs)
    distance = np.arange(length) * RESOLUTION

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    means = {}
    for name, style in SERIES.items():
        means[name] = mean_curve(curves[name], length)
        ax.plot(distance, means[name], color=style["color"], linestyle="--",
                linewidth=1.8, label=style["label"])

    final_values = {name: means[name][-1] for name in SERIES}
    label_y = spread_labels(final_values, min_gap=3.3)
    for name in SERIES:
        ax.annotate(f"{final_values[name]:.1f}%", xy=(distance[-1], label_y[name]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK_SECONDARY, annotation_clip=False)

    ax.set_xlim(0, distance[-1])
    ax.set_ylim(0, 100)
    ax.set_xlabel("Distanza percorsa (m)", color=INK_PRIMARY)
    ax.set_ylabel("Copertura (%)", color=INK_PRIMARY)
    style_axes(ax, "y")
    # tacche senza numero a meta' tra i valori etichettati (50, 150, ...)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.tick_params(axis="x", which="minor", length=3.5, color=INK_SECONDARY)

    legend = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_BASE}.pdf")
    fig.savefig(f"{OUTPUT_BASE}.png", dpi=200)
    plt.close(fig)
    print(f"Grafico salvato in {OUTPUT_BASE}.pdf e .png")

    print("\nCopertura media a varie distanze:")
    for d in (25, 50, 100, 150, 200, 300, 400):
        i = int(round(d / RESOLUTION))
        print(f"  {d:3d} m | " + " | ".join(f"{name} {means[name][i]:5.1f}%" for name in SERIES))
    print("\nDistanza alla quale la copertura media raggiunge:")
    for target in (50, 80, 85, 90):
        reached = []
        for name in SERIES:
            idx = np.argmax(means[name] >= target) if means[name].max() >= target else None
            reached.append(f"{name} " + (f"{idx * RESOLUTION:5.1f} m" if idx is not None else "  mai"))
        print(f"  {target}% | " + " | ".join(reached))

    plot_final_coverage()


def plot_final_coverage():
    with open(RUNS_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    names = list(SERIES)[::-1]   # dal basso verso l'alto, cosi' Random compare in cima
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    rng = np.random.default_rng(0)   # spostamento verticale fisso, per separare i punti sovrapposti

    print("\nRun bloccati (copertura finale sotto il 90%):")
    for y, name in enumerate(names):
        runs = [r for r in rows if r["strategy"] == name]
        coverage = np.array([float(r["coverage_pct"]) for r in runs])
        jitter = rng.uniform(-0.18, 0.18, len(runs))
        ax.scatter(coverage, y + jitter, s=34, color=SERIES[name]["color"],
                   edgecolors="white", linewidths=0.8, zorder=3)

        blocked = [r for r in runs if float(r["coverage_pct"]) < BLOCKED_THRESHOLD]
        if not blocked:
            count_text = f"nessun blocco su {len(runs)}"
        else:
            count_text = f"{len(blocked)} {'bloccato' if len(blocked) == 1 else 'bloccati'} su {len(runs)}"
        ax.text(101.5, y, count_text, va="center", fontsize=9, color=INK_PRIMARY, clip_on=False)
        print(f"  {name:10s} {len(blocked):2d} | " +
              ", ".join(f"seed {r['seed']} ({float(r['coverage_pct']):.1f}%)" for r in blocked))

    ax.axvline(BLOCKED_THRESHOLD, color=AXIS, linewidth=1, zorder=1)
    ax.text(BLOCKED_THRESHOLD, len(names) - 0.45, "90%", ha="center", va="bottom",
            fontsize=8, color=INK_SECONDARY)

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([SERIES[n]["label"] for n in names], color=INK_PRIMARY)
    ax.set_xlabel("Copertura finale del run (%)", color=INK_PRIMARY)
    style_axes(ax, "x")
    ax.tick_params(axis="y", labelcolor=INK_PRIMARY, labelsize=10)

    fig.tight_layout()
    fig.savefig(f"{FINAL_OUTPUT_BASE}.pdf")
    fig.savefig(f"{FINAL_OUTPUT_BASE}.png", dpi=200)
    plt.close(fig)
    print(f"Grafico salvato in {FINAL_OUTPUT_BASE}.pdf e .png")


if __name__ == "__main__":
    main()
