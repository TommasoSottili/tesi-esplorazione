
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from forest_generator import generate_forest
from lidar import Lidar
from local_grid import build_local_grid, UNKNOWN, FREE, OCCUPIED

# Bosco, robot, scansione (come nel test del lidar)
forest = generate_forest(20, 20, n_trees=15, n_rocks=3, seed=42)
robot_state = (10.0, 10, 0.0)
sensor = Lidar(n_rays=360, r_max=6.0)
points = np.array(sensor.scan(robot_state, forest))

# Costruisco la griglia locale dalla scansione
grid = build_local_grid(robot_state, points, window_size=4.0, resolution=0.2, r_max=6.0)
# informazioni della griglia
print("Forma della griglia:", grid.shape)
print("Valori presenti:", np.unique(grid))
print("Celle UNKNOWN (grigio):", np.sum(grid == UNKNOWN))
print("Celle FREE (bianco):   ", np.sum(grid == FREE))
print("Celle OCCUPIED (nero): ", np.sum(grid == OCCUPIED))
print("Totale celle:", grid.size)
fig, (ax_world, ax_grid) = plt.subplots(1, 2, figsize=(16, 8))

rx, ry, _ = robot_state          # estraggo la posizione vera del robot
half = 4.0 / 2                 

forest.plot(ax_world)
ax_world.plot(rx, ry, 'r*', markersize=15)                    # robot
ax_world.plot(points[:, 0], points[:, 1], 'b.', markersize=3) # scansione
# riquadro 4x4 centrato sul robot
ax_world.plot([rx-half, rx+half, rx+half, rx-half, rx-half],
              [ry-half, ry-half, ry+half, ry+half, ry-half], 'r-', linewidth=1.5)
ax_world.set_title("Bosco reale + scansione (riquadro = finestra 4x4)")
# colori: sconosciuto=grigio, libero=bianco, occupato=nero
colormap = ListedColormap(['gray', 'white', 'black'])
ax_grid.imshow(grid, cmap=colormap, origin='lower', vmin=0, vmax=2)
ax_grid.set_title("Griglia locale (grigio=ignoto, bianco=libero, nero=occupato)")
plt.show()