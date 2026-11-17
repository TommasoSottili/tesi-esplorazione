
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Patch, Polygon as PolygonPatch
from shapely.geometry import Point
from forest_generator import generate_forest
from lidar import Lidar
from local_grid import build_local_grid
from global_map import GlobalMap, FREE
from plot_forest_example import SEED, START, FIGURE_DIR, INK_PRIMARY, INK_SECONDARY, AXIS, save

R_MAX = 4.0
RESOLUTION = 0.2
WINDOW_SIZE = 8.0
CORONA_MARGIN = 0.6
INFLATION_RADIUS = 0.2
ZOOM = 4.6  

OUTPUT_BASE = os.path.join(FIGURE_DIR, "mappa_prima_scansione")

STATE_COLORS = ["gray", "white", "black", "red"]
STATE_LABELS = ["sconosciuta", "libera", "occupata", "di sicurezza"]


def main():
    forest = generate_forest(20, 20, n_trees=15, n_rocks=3, seed=SEED)
    sensor = Lidar(n_rays=360, r_max=R_MAX)
    robot_state = (START[0], START[1], 0.0)

    scan_points = np.array(sensor.scan(robot_state, forest))
    global_map = GlobalMap(20.0, 20.0, RESOLUTION)
    local = build_local_grid(robot_state, scan_points, WINDOW_SIZE, RESOLUTION, R_MAX,
                             corona_margin=CORONA_MARGIN)
    global_map.update(local, robot_state, WINDOW_SIZE)
    robot_cell = global_map.world_to_cell(*START)
    global_map.grid[robot_cell[0]][robot_cell[1]] = FREE
    global_map.inflate_obstacles(INFLATION_RADIUS, robot_cell=robot_cell)

    fig, ax = plt.subplots(figsize=(4.6, 4.9))
    ax.imshow(global_map.grid, cmap=ListedColormap(STATE_COLORS), origin="lower", vmin=0, vmax=3,
              extent=[0, 20, 0, 20], interpolation="nearest")

    hits = scan_points[np.hypot(scan_points[:, 0] - START[0], scan_points[:, 1] - START[1]) < R_MAX - 1e-6]
    hit_obstacles = [o for o in forest.obstacles if any(o.distance(Point(p)) < 1e-6 for p in hits)]
    for obstacle in hit_obstacles:
        ax.add_patch(PolygonPatch(list(obstacle.exterior.coords), closed=True, fill=False,
                                  edgecolor="#2a78d6", linewidth=0.9))
    ax.add_patch(Circle(START, R_MAX, fill=False, edgecolor=INK_PRIMARY, linewidth=0.9, linestyle="--"))
    ax.plot(*START, marker="o", markersize=6, color="#2a78d6", markeredgecolor="white", zorder=3)

    ax.set_xlim(START[0] - ZOOM, START[0] + ZOOM)
    ax.set_ylim(START[1] - ZOOM, START[1] + ZOOM)
    ax.set_aspect("equal")
    ax.set_xticks(range(6, 15, 2))
    ax.set_yticks(range(6, 15, 2))
    ax.set_xlabel("x (m)", color=INK_PRIMARY)
    ax.set_ylabel("y (m)", color=INK_PRIMARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    for side in ax.spines.values():
        side.set_color(AXIS)

    handles = [Patch(facecolor=c, edgecolor=INK_SECONDARY, linewidth=0.6, label=l)
               for c, l in zip(STATE_COLORS, STATE_LABELS)]
    legend = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4,
                       frameon=False, fontsize=8, handlelength=1.2, columnspacing=1.2)
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)

    save(fig, OUTPUT_BASE)
    counts = {l: int(np.sum(global_map.grid == k)) for k, l in enumerate(STATE_LABELS)}
    print(counts)


if __name__ == "__main__":
    main()
