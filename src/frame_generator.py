
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from forest_generator import generate_forest
from lidar import Lidar
from exploration import run_exploration
from strategies import random_strategy, frontier_strategy, serpentine_strategy
from render import render_frame

SEED = 1
STRATEGY_NAME = "serpentine"   # "random" | "frontier" | "serpentine"
CORONA_MARGIN = 0.0           

STRATEGIES = {
    "random": random_strategy,
    "frontier": frontier_strategy,
    "serpentine": serpentine_strategy,
}

WORLD_WIDTH = 20.0
WORLD_HEIGHT = 20.0
RESOLUTION = 0.2

OUTPUT_DIR = os.path.join("..", "output", f"film_{STRATEGY_NAME}_seed{SEED}_corona{CORONA_MARGIN}")


forest = generate_forest(20, 20, n_trees=15, n_rocks=3, seed=SEED)
sensor = Lidar(n_rays=360, r_max=4.0)
strategy = STRATEGIES[STRATEGY_NAME]

global_map, point_cloud, stats = run_exploration(
    forest, sensor, (10.0, 10.0, 0.0), strategy,
    world_width=WORLD_WIDTH, world_height=WORLD_HEIGHT, resolution=RESOLUTION,
    window_size=8.0, inflation_radius=0.2, max_steps=300,
    corona_margin=CORONA_MARGIN
)

trajectory = stats["trajectory"]
grid_history = stats["grid_history"]
point_cloud_history = stats["point_cloud_history"]
n_frames = len(trajectory)

os.makedirs(OUTPUT_DIR, exist_ok=True)

frames = []
accumulated_points = np.empty((0, 2))
for i in range(n_frames):
    if len(point_cloud_history[i]) > 0:
        accumulated_points = np.vstack([accumulated_points, point_cloud_history[i]])

    fig = render_frame(grid_history[i], trajectory[:i + 1],
                        WORLD_WIDTH, WORLD_HEIGHT, RESOLUTION, i, n_frames,
                        point_cloud_so_far=accumulated_points)
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert('RGB')
    frames.append(img)
    plt.close(fig)

print(f"Generati {n_frames} frame in memoria")

gif_path = os.path.join(OUTPUT_DIR, "film.gif")
frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)

print(f"Gif salvato in {gif_path}")
