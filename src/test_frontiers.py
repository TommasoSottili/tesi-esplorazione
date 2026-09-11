
import numpy as np
import matplotlib.pyplot as plt
from forest_generator import generate_forest
from lidar import Lidar
from local_grid import build_local_grid
from global_map import GlobalMap

forest = generate_forest(20, 20, n_trees=15, n_rocks=3, seed=42)
sensor = Lidar(n_rays=360, r_max=4.0)
global_map = GlobalMap(world_width=20, world_height=20, resolution=0.2)

positions = [
    (10.0, 10.0, 0.0),
    (11.0, 6.5, 0.0),
]
for robot_state in positions:
    points = np.array(sensor.scan(robot_state, forest))
    local = build_local_grid(robot_state, points, window_size=8.0, resolution=0.2, r_max=4.0)
    global_map.update(local, robot_state, window_size=8.0)

global_map.inflate_obstacles(inflation_radius=0.2)

frontiers = global_map.find_frontiers()
print(f"Numero di celle di frontiera: {len(frontiers)}")

fig, ax = plt.subplots(figsize=(9, 9))
global_map.plot(ax)

if frontiers:
    fx = [col * global_map.resolution for (row, col) in frontiers]
    fy = [row * global_map.resolution for (row, col) in frontiers]
    ax.plot(fx, fy, 'c.', markersize=4)   # 'c.' = puntini ciano

for robot_state in positions:
    ax.plot(robot_state[0], robot_state[1], 'r*', markersize=12)

ax.set_title("Frontiere (ciano) = confine tra noto e ignoto")
plt.show()