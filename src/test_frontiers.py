
import numpy as np
import matplotlib.pyplot as plt
from forest_generator import generate_forest
from lidar import Lidar
from local_grid import build_local_grid
from global_map import GlobalMap

# 1. Bosco, sensore, mappa
forest = generate_forest(20, 20, n_trees=15, n_rocks=3, seed=42)
sensor = Lidar(n_rays=360, r_max=6.0)
global_map = GlobalMap(world_width=20, world_height=20, resolution=0.2)

# 2. Esploro POCO (poche posizioni), così resta molto ignoto e le frontiere si vedono
positions = [
    (10.0, 10.0, 0.0),
    (11.0, 6.5, 0.0),
]
for robot_state in positions:
    points = np.array(sensor.scan(robot_state, forest))
    local = build_local_grid(robot_state, points, window_size=4.0, resolution=0.2, r_max=6.0)
    global_map.update(local, robot_state, window_size=4.0)

global_map.inflate_obstacles(inflation_radius=0.4)

# 3. Trovo le frontiere
frontiers = global_map.find_frontiers()
print(f"Numero di celle di frontiera: {len(frontiers)}")

# 4. Disegno la mappa con le frontiere sopra
fig, ax = plt.subplots(figsize=(9, 9))
global_map.plot(ax)

# traduco le celle di frontiera in metri e le disegno come puntini
if frontiers:
    fx = [col * global_map.resolution for (row, col) in frontiers]
    fy = [row * global_map.resolution for (row, col) in frontiers]
    ax.plot(fx, fy, 'c.', markersize=4)   # 'c.' = puntini ciano

# segno le posizioni del robot
for robot_state in positions:
    ax.plot(robot_state[0], robot_state[1], 'r*', markersize=12)

ax.set_title("Frontiere (ciano) = confine tra noto e ignoto")
plt.show()