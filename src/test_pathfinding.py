
import numpy as np
import matplotlib.pyplot as plt
from forest_generator import generate_forest
from lidar import Lidar
from local_grid import build_local_grid
from global_map import GlobalMap
from pathfinding import find_path

# 1. Bosco, sensore, mappa globale
forest = generate_forest(20, 20, n_trees=15, n_rocks=3, seed=42)
sensor = Lidar(n_rays=360, r_max=4.0)
global_map = GlobalMap(world_width=20, world_height=20, resolution=0.2)

# 2. Scansiono da una griglia di posizioni per creare un'ampia zona nota e connessa
for x in range(3, 18, 2):
    for y in range(3, 18, 2):
        robot_state = (float(x), float(y), 0.0)
        points = np.array(sensor.scan(robot_state, forest))
        local = build_local_grid(robot_state, points, window_size=8.0, resolution=0.2, r_max=4.0)
        global_map.update(local, robot_state, window_size=8.0)

# 3. Gonfio gli ostacoli (il percorso dovrà evitare anche le zone rosse)
global_map.inflate_obstacles(inflation_radius=0.4)

# 4. Scelgo partenza e arrivo IN METRI, poi li traduco in celle
start_world = (8.0, 1.0)
goal_world = (16.0, 16.0)

start_cell = global_map.world_to_cell(*start_world)
goal_cell = global_map.world_to_cell(*goal_world)
print("Cella di partenza:", start_cell)
print("Cella di arrivo:  ", goal_cell)

# 5. Calcolo il percorso con Dijkstra
path = find_path(global_map, start_cell, goal_cell)

if path is None:
    print("Nessun percorso trovato!")
else:
    print(f"Percorso trovato: {len(path)} celle")

# 6. Disegno la mappa con il percorso sopra
fig, ax = plt.subplots(figsize=(9, 9))
global_map.plot(ax)

if path is not None:
    # traduco le celle del percorso in coordinate del mondo (metri) per disegnarle
    path_x = [(col + 0.5) * global_map.resolution for (row, col) in path]
    path_y = [(row + 0.5) * global_map.resolution for (row, col) in path]
    ax.plot(path_x, path_y, 'b-', linewidth=2)            # la linea del percorso
    ax.plot(start_world[0], start_world[1], 'go', markersize=12)  # partenza (verde)
    ax.plot(goal_world[0], goal_world[1], 'mo', markersize=12)    # arrivo (magenta)

plt.show()