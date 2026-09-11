
import random
import numpy as np


def random_strategy(global_map, frontiers, robot_cell, context):

    if not frontiers:
        return None

    return random.choice(frontiers)


def frontier_strategy(global_map, frontiers, robot_cell, context):


    if not frontiers:
        return None

    distances = context["distances"]
    min_distance = min(distances[f] for f in frontiers)
    nearest_frontiers = [f for f in frontiers if distances[f] == min_distance]

    return random.choice(nearest_frontiers)


def compute_serpentine_rank(n_rows, n_cols, resolution, macro_cell_size=1.0):
   
    # Restituisce comunque un array sulla griglia fine (n_rows x n_cols), ogni
    # cella fine porta il rango della macro-cella che la contiene. Così
    # serpentine_strategy resta invariata
    cells_per_macro = int(round(macro_cell_size / resolution))
    if cells_per_macro < 1:
        cells_per_macro = 1

   
    n_macro_rows = (n_rows + cells_per_macro - 1) // cells_per_macro
    n_macro_cols = (n_cols + cells_per_macro - 1) // cells_per_macro

    macro_rank = np.zeros((n_macro_rows, n_macro_cols), dtype=int)
    for mr in range(n_macro_rows):
        for mc in range(n_macro_cols):
            if mr % 2 == 0:
                macro_rank[mr][mc] = mr * n_macro_cols + mc
            else:
                macro_rank[mr][mc] = mr * n_macro_cols + (n_macro_cols - 1 - mc)

   
    rank = np.zeros((n_rows, n_cols), dtype=int)
    for row in range(n_rows):
        for col in range(n_cols):
            rank[row][col] = macro_rank[row // cells_per_macro][col // cells_per_macro]
    return rank


def serpentine_strategy(global_map, frontiers, robot_cell, context):

    if not frontiers:
        return None

    rank = context["rank"]
    min_rank = min(rank[f[0]][f[1]] for f in frontiers)
    lowest_rank_frontiers = [f for f in frontiers if rank[f[0]][f[1]] == min_rank]

    return random.choice(lowest_rank_frontiers)
