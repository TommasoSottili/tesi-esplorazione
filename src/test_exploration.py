
import numpy as np
import matplotlib.pyplot as plt
from forest_generator import generate_forest
from lidar import Lidar
from exploration import run_exploration
from strategies import random_strategy

# 1. Genero il bosco (il robot partirà dal centro di default, coerente con la clearance)
forest = generate_forest(20, 20, n_trees=15, n_rocks=3, seed=42)
sensor = Lidar(n_rays=360, r_max=4.0)
start_position = (10.0, 10.0, 0.0)   # stesso centro protetto dalla clearance

# 2. Lancio l'esplorazione con la strategia casuale
global_map, point_cloud, stats = run_exploration(
    forest, sensor, start_position, random_strategy,
    world_width=20.0, world_height=20.0, resolution=0.2,
    window_size=8.0, inflation_radius=0.4, max_steps=300
)

# 3. Stampo le statistiche
print("Passi effettuati:", stats["steps"])
print("Distanza percorsa: %.2f m" % stats["distance"])
print("Celle conosciute:", stats["cells_known"], "su", global_map.n_rows * global_map.n_cols)
print("Punti nella point cloud:", len(point_cloud))

# 4. Disegno: bosco vero a sinistra, mappa esplorata + point cloud a destra
fig, (ax_world, ax_map) = plt.subplots(1, 2, figsize=(16, 8))

forest.plot(ax_world)
ax_world.set_title("Bosco reale (ground truth)")

global_map.plot(ax_map)
if len(point_cloud) > 0:
    ax_map.plot(point_cloud[:, 0], point_cloud[:, 1], 'c.', markersize=2)
ax_map.set_title(f"Mappa esplorata (strategia casuale) - {stats['steps']} passi")

plt.show()