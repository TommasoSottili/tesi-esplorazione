import matplotlib.pyplot as plt
from shapely.geometry import Point
from forest_generator import generate_forest

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

forest_a = generate_forest(20, 20, n_trees=15, n_rocks=3, seed=42)
forest_a.plot(axes[0])
axes[0].plot(10, 10, 'r*', markersize=15)  
axes[0].set_title("Bosco A (seed=42)")

forest_b = generate_forest(20, 20, n_trees=15, n_rocks=3, seed=42)
forest_b.plot(axes[1])
axes[1].plot(10, 10, 'r*', markersize=15)
axes[1].set_title("Bosco B (seed=42, identico ad A)")

forest_c = generate_forest(20, 20, n_trees=15, n_rocks=3, seed=99)
forest_c.plot(axes[2])
axes[2].plot(10, 10, 'r*', markersize=15)
axes[2].set_title("Bosco C (seed=99, diverso)")

plt.show()