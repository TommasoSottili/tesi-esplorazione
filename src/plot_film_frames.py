
import os
import numpy as np
from shapely.geometry import Point
from forest_generator import generate_forest
from lidar import Lidar
from exploration import run_exploration
from strategies import serpentine_strategy
from render import render_frame
from plot_forest_example import FIGURE_DIR, save

SEED = 21
START = (0.1, 0.1, 0.0)        
MACRO_CELL_SIZE = 4.0
WORLD_SIZE = 20.0
RESOLUTION = 0.2

OUTPUT_NAMES = ("fotogramma_serpentina_meta", "fotogramma_serpentina_finale")


def main():
    forest = generate_forest(20, 20, n_trees=15, n_rocks=3, seed=SEED,
                             start_point=Point(START[0], START[1]))
    _, _, stats = run_exploration(forest, Lidar(n_rays=360, r_max=4.0), START, serpentine_strategy,
                                  world_width=WORLD_SIZE, world_height=WORLD_SIZE, resolution=RESOLUTION,
                                  macro_cell_size=MACRO_CELL_SIZE)
    n_frames = len(stats["trajectory"])
    print(f"Fotogrammi: {n_frames} | distanza {stats['distance']:.1f} m | "
          f"copertura {100 * stats['cells_known'] / 10000:.1f}%")

    for index, name in zip((n_frames // 2, n_frames - 1), OUTPUT_NAMES):
        points = np.vstack(stats["point_cloud_history"][:index + 1])
        fig = render_frame(stats["grid_history"][index], stats["trajectory"][:index + 1],
                           WORLD_SIZE, WORLD_SIZE, RESOLUTION, index, n_frames,
                           point_cloud_so_far=points)
        fig.axes[0].lines[0].set_rasterized(True)
        fig.set_dpi(300)
        save(fig, os.path.join(FIGURE_DIR, name))


if __name__ == "__main__":
    main()
