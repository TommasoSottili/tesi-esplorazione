
import numpy as np
from global_map import GlobalMap, FREE
from local_grid import build_local_grid
from pathfinding import find_path


def run_exploration(forest, sensor, start_position, strategy,
                     world_width=20.0, world_height=20.0, resolution=0.2,
                     window_size=8.0, inflation_radius=0.4, max_steps=300):

    robot_state = start_position
    global_map = GlobalMap(world_width, world_height, resolution)
    point_cloud_parts = []
    total_distance = 0.0
    steps = 0
    coverage_history = []   # celle note dopo ogni step, per confrontare strategie diverse

    while steps < max_steps:

        # --- 1. PERCEZIONE ---
        scan_points = np.array(sensor.scan(robot_state, forest))
        robot_x, robot_y, _ = robot_state

        # separo gli impatti REALI (per la point cloud) dai raggi a vuoto (r_max)
        dists = np.hypot(scan_points[:, 0] - robot_x, scan_points[:, 1] - robot_y)
        hits = scan_points[dists < sensor.r_max - 1e-6]
        if len(hits) > 0:
            point_cloud_parts.append(hits)

        # --- 2. AGGIORNAMENTO MAPPA ---
        robot_cell = global_map.world_to_cell(robot_x, robot_y)
        local = build_local_grid(robot_state, scan_points, window_size, resolution, sensor.r_max)
        global_map.update(local, robot_state, window_size)

        # il robot occupa fisicamente questa cella senza essere entrato in
        # collisione: non può essere un ostacolo, anche se una scansione
        # ravvicinata la marca cosi' per un lembo di ostacolo che taglia
        # solo un angolo della cella (il centro resta comunque libero)
        if robot_cell is not None:
            global_map.grid[robot_cell[0]][robot_cell[1]] = FREE

        global_map.inflate_obstacles(inflation_radius, robot_cell=robot_cell)
        coverage_history.append(int(np.sum(global_map.grid != 0)))

        # --- 3. FRONTIERE E CONDIZIONE DI STOP ---
        frontiers = global_map.find_frontiers()
        if not frontiers:
            break   # niente più da esplorare: esplorazione completata
                # --- 4. DECISIONE: la strategia sceglie dove andare ---
        target_cell = strategy(global_map, frontiers, robot_cell)

        if target_cell is None:
            break   # la strategia non sa scegliere: mi fermo per sicurezza

        # --- 5. PATHFINDING: calcolo il percorso verso la destinazione scelta ---
        path = find_path(global_map, robot_cell, target_cell)

        if path is None or len(path) < 2:
            # la frontiera scelta non è raggiungibile: la rimuovo e ci fermiamo
            # per questo passo (la prossima iterazione ne troverà altre)
            frontiers.remove(target_cell)
            steps += 1
            continue

        # --- 6. MOVIMENTO: sposto il robot alla fine del percorso trovato ---
        goal_row, goal_col = path[-1]
        new_x = goal_col * resolution + resolution / 2
        new_y = goal_row * resolution + resolution / 2

        total_distance += np.hypot(new_x - robot_x, new_y - robot_y)
        robot_state = (new_x, new_y, 0.0)

        steps += 1

    # --- 7. Risultati finali ---
    point_cloud = np.vstack(point_cloud_parts) if point_cloud_parts else np.empty((0, 2))
    stats = {
        "steps": steps,
        "distance": total_distance,
        "cells_known": int(np.sum(global_map.grid != 0)),
        "coverage_history": coverage_history,
    }
    return global_map, point_cloud, stats