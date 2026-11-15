
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon as PolygonPatch
from forest_generator import generate_forest

SEED = 1
START = (10.0, 10.0)
CLEARANCE = 1.5

FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "relazione", "figure")
OUTPUT_BASE = os.path.join(FIGURE_DIR, "bosco_esempio")

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
AXIS = "#c3c2b7"
OBSTACLE_FILL = "#a3a29b"


def draw_forest(ax, forest):
    for obstacle in forest.obstacles:
        ax.add_patch(PolygonPatch(list(obstacle.exterior.coords), closed=True,
                                  facecolor=OBSTACLE_FILL, edgecolor=INK_SECONDARY, linewidth=0.8))


def style_map_axes(ax):
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, 21, 5))
    ax.set_yticks(range(0, 21, 5))
    ax.set_xlabel("x (m)", color=INK_PRIMARY)
    ax.set_ylabel("y (m)", color=INK_PRIMARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    for side in ax.spines.values():
        side.set_color(AXIS)


def save(fig, output_base):
    fig.tight_layout()
    fig.savefig(f"{output_base}.pdf")
    fig.savefig(f"{output_base}.png", dpi=200)
    plt.close(fig)
    print(f"Figura salvata in {output_base}.pdf e .png")


def main():
    forest = generate_forest(20, 20, n_trees=15, n_rocks=3, seed=SEED)

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    draw_forest(ax, forest)

    ax.add_patch(Circle(START, CLEARANCE, fill=False, edgecolor=INK_SECONDARY,
                        linewidth=0.9, linestyle="--"))
    ax.plot(*START, marker="o", markersize=6, color=INK_PRIMARY)
    ax.annotate("partenza", xy=START, xytext=(25, 0), textcoords="offset points",
                ha="left", va="center", fontsize=8, color=INK_PRIMARY)

    style_map_axes(ax)
    save(fig, OUTPUT_BASE)


if __name__ == "__main__":
    main()
