
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from global_map import GlobalMap


def cell_to_world(cell, resolution):
    row, col = cell
    x = col * resolution + resolution / 2
    y = row * resolution + resolution / 2
    return x, y


def render_frame(grid, trajectory_so_far, world_width, world_height, resolution, step_index, total_steps,
                  point_cloud_so_far=None):

    fig, ax = plt.subplots(figsize=(6, 6))

    gm = GlobalMap(world_width, world_height, resolution)
    gm.grid = grid
    gm.plot(ax)

    if point_cloud_so_far is not None and len(point_cloud_so_far) > 0:
        ax.plot(point_cloud_so_far[:, 0], point_cloud_so_far[:, 1], 'c.', markersize=2)

    xs, ys = zip(*[cell_to_world(c, resolution) for c in trajectory_so_far])
    ax.plot(xs, ys, '-', color='blue', linewidth=1.2)
    ax.plot(xs[-1], ys[-1], 'o', color='orange', markersize=7, markeredgecolor='black')

    ax.set_title(f"Esplorazione - step {step_index + 1}/{total_steps}")

    return fig
