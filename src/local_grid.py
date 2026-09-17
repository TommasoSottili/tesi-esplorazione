
import numpy as np

UNKNOWN = 0
FREE = 1
OCCUPIED = 2


def world_to_cell(wx, wy, robot_x, robot_y, window_size, resolution):
    
    dx = wx - robot_x
    dy = wy - robot_y

    half = window_size / 2.0

   
    col = int((dx + half) / resolution + 1e-9)
    row = int((dy + half) / resolution + 1e-9)

    n_cells = int(window_size / resolution)

    if row < 0 or row >= n_cells or col < 0 or col >= n_cells:
        return None

    return (row, col)

def build_local_grid(robot_state, scan_points, window_size=4.0, resolution=0.2, r_max=2.0, free_margin=None, corona_margin=None):

    robot_x, robot_y, _ = robot_state


    if free_margin is None:
        free_margin = resolution

    if corona_margin is None:
        corona_margin = resolution

    # Creo la griglia, inizialmente tutta sconosciuta
    n_cells = int(window_size / resolution)
    grid = np.full((n_cells, n_cells), UNKNOWN, dtype=int)

    # Per ogni punto della scansione
    for point in scan_points:
        px, py = point[0], point[1]

        # distanza del punto dal robot (distanza euclidea)
        distance = np.hypot(px - robot_x, py - robot_y)
        if distance < 1e-9:
            continue

        is_hit = distance < r_max - 1e-6


        free_distance = max(distance - free_margin, 0.0) if is_hit else max(distance - corona_margin, 0.0)

        dir_x = (px - robot_x) / distance
        dir_y = (py - robot_y) / distance

        n_samples = int(free_distance / (resolution / 4))  # un campione ogni quarto di cella
        for i in range(n_samples):
            t = i / n_samples       
            sample_x = robot_x + t * free_distance * dir_x
            sample_y = robot_y + t * free_distance * dir_y

            cell = world_to_cell(sample_x, sample_y, robot_x, robot_y, window_size, resolution)
            if cell is not None:
                row, col = cell
                grid[row][col] = FREE

        if is_hit:
            cell = world_to_cell(px, py, robot_x, robot_y, window_size, resolution)
            if cell is not None:
                row, col = cell
                grid[row][col] = OCCUPIED

    return grid