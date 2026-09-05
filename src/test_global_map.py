
import numpy as np
import matplotlib.pyplot as plt
from forest_generator import generate_forest
from lidar import Lidar
from local_grid import build_local_grid
from global_map import GlobalMap, UNKNOWN, FREE, OCCUPIED, INFLATED

# 1. Bosco e sensore
forest = generate_forest(20, 20, n_trees=15, n_rocks=3, seed=42)
sensor = Lidar(n_rays=360, r_max=4.0)

# 2. Mappa globale vuota
global_map = GlobalMap(world_width=20, world_height=20, resolution=0.2)

# 3. Posizioni da cui il robot scansiona
positions = [
    (10.0, 10.0, 0.0),
    (6.0, 6.0, 0.0),
    (14.0, 13.0, 0.0),
]

# 4. Fusione delle scansioni
for robot_state in positions:
    points = np.array(sensor.scan(robot_state, forest))
    local = build_local_grid(robot_state, points, window_size=8.0, resolution=0.2, r_max=4.0)
    global_map.update(local, robot_state, window_size=8.0)

# 5. NOVITÀ: gonfio gli ostacoli (fascia di sicurezza)
global_map.inflate_obstacles(inflation_radius=0.4)

# debug: conteggio dei quattro stati
print("Celle UNKNOWN: ", np.sum(global_map.grid == UNKNOWN))
print("Celle FREE:    ", np.sum(global_map.grid == FREE))
print("Celle OCCUPIED:", np.sum(global_map.grid == OCCUPIED))
print("Celle INFLATED:", np.sum(global_map.grid == INFLATED))

# 6. Disegno affiancato
fig, (ax_world, ax_map) = plt.subplots(1, 2, figsize=(16, 8))

forest.plot(ax_world)
for robot_state in positions:
    ax_world.plot(robot_state[0], robot_state[1], 'r*', markersize=12)
ax_world.set_title("Bosco reale (stelle = posizioni del robot)")

global_map.plot(ax_map)

plt.show()