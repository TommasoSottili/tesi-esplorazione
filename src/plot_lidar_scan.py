
import os
import numpy as np
import matplotlib.pyplot as plt
from forest_generator import generate_forest
from lidar import Lidar
from plot_forest_example import (SEED, START, FIGURE_DIR, INK_PRIMARY, INK_SECONDARY,
                                 draw_forest, style_map_axes, save)

R_MAX = 4.0
OUTPUT_BASE = os.path.join(FIGURE_DIR, "scansione_lidar")

HIT_COLOR = "#2a78d6"
NO_HIT_COLOR = "#d6d5ce"


def main():
    forest = generate_forest(20, 20, n_trees=15, n_rocks=3, seed=SEED)
    sensor = Lidar(n_rays=360, r_max=R_MAX)
    points, ranges = sensor.scan((START[0], START[1], 0.0), forest, return_ranges=True)
    is_hit = ranges < R_MAX - 1e-6

    fig, ax = plt.subplots(figsize=(4.6, 4.9))
    draw_forest(ax, forest)

    for (px, py), hit in zip(points, is_hit):
        ax.plot([START[0], px], [START[1], py], color=HIT_COLOR if hit else NO_HIT_COLOR,
                linewidth=0.6, zorder=1 if not hit else 2)
    ax.plot(points[is_hit, 0], points[is_hit, 1], linestyle="none", marker="o", markersize=2.5,
            color=HIT_COLOR, zorder=3)
    ax.plot(*START, marker="o", markersize=6, color=INK_PRIMARY, zorder=4)

    ax.plot([], [], color=HIT_COLOR, linewidth=1.5, marker="o", markersize=3,
            label=f"raggi con impatto ({int(is_hit.sum())})")
    ax.plot([], [], color=NO_HIT_COLOR, linewidth=1.5,
            label=f"raggi senza impatto ({int((~is_hit).sum())})")
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
                       frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)

    style_map_axes(ax)
    save(fig, OUTPUT_BASE)
    print(f"raggi con impatto: {int(is_hit.sum())} su {len(ranges)}")


if __name__ == "__main__":
    main()
