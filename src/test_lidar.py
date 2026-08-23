

import numpy as np
import matplotlib.pyplot as plt
from forest_generator import generate_forest
from lidar import Lidar

forest = generate_forest(20, 20, n_trees=15, n_rocks=3, seed=42)

robot_x, robot_y, robot_theta = 10.0, 10.0, 0.0
robot_state = (robot_x, robot_y, robot_theta)

sensor = Lidar(n_rays=360, r_max=6.0)
points = sensor.scan(robot_state, forest)

fig, ax = plt.subplots(figsize=(8, 8))
forest.plot(ax)                                    
ax.plot(robot_x, robot_y, 'r*', markersize=15)     

points = np.array(points)
ax.plot(points[:, 0], points[:, 1], 'b.', markersize=3)

ax.set_title("Scansione LiDAR dal centro del bosco")
plt.show()