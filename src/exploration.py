
import numpy as np
from global_map import GlobalMap, FREE
from local_grid import build_local_grid
from pathfinding import find_path, reachable_cells_with_distance
from strategies import compute_serpentine_rank


def run_exploration(forest, sensor, start_position, strategy,
                     world_width=20.0, world_height=20.0, resolution=0.2,
                     window_size=8.0, inflation_radius=0.2, max_steps=300,
                     macro_cell_size=1.0, corona_margin=0.6):
    
    robot_state = start_position
    global_map = GlobalMap(world_width, world_height, resolution)
    point_cloud_parts = []
    total_distance = 0.0
    steps = 0                 
    coverage_history = []      
    trajectory = []             
    stuck_frontiers = set()   
    pending_check = None       

    rank = compute_serpentine_rank(global_map.n_rows, global_map.n_cols,
                                    resolution, macro_cell_size)

    
    path = None
   
    scan_needed = True

    while steps < max_steps:

        if scan_needed:
            scan_points = np.array(sensor.scan(robot_state, forest))
            robot_x, robot_y, _ = robot_state

            dists = np.hypot(scan_points[:, 0] - robot_x, scan_points[:, 1] - robot_y)
            hits = scan_points[dists < sensor.r_max - 1e-6]
            if len(hits) > 0:
                point_cloud_parts.append(hits)

            robot_cell = global_map.world_to_cell(robot_x, robot_y)
            local = build_local_grid(robot_state, scan_points, window_size, resolution, sensor.r_max,
                                      corona_margin=corona_margin)
            global_map.update(local, robot_state, window_size)

            
            if robot_cell is not None:
                global_map.grid[robot_cell[0]][robot_cell[1]] = FREE

            global_map.inflate_obstacles(inflation_radius, robot_cell=robot_cell)
            coverage_history.append(int(np.sum(global_map.grid != 0)))
            trajectory.append(robot_cell)
            scan_needed = False

        if path is None:
            frontiers = global_map.find_frontiers()
            if not frontiers:
                break   # niente più da esplorare: esplorazione completata

            
            if pending_check is not None and pending_check in frontiers:
                stuck_frontiers.add(pending_check)
            pending_check = None

            
            distances = reachable_cells_with_distance(global_map, robot_cell)
            frontiers = [f for f in frontiers if f in distances and f != robot_cell and f not in stuck_frontiers]
            if not frontiers:
                break   # nessuna frontiera nota è raggiungibile (o utile): mi fermo per sicurezza

            context = {"distances": distances, "rank": rank}
            target_cell = strategy(global_map, frontiers, robot_cell, context)

            if target_cell is None:
                break  

            path = find_path(global_map, robot_cell, target_cell)

            if path is None or len(path) < 2:
               
                stuck_frontiers.add(target_cell)
                path = None
                steps += 1
                continue

            steps += 1

        next_cell = path[1]

        if global_map.grid[next_cell[0]][next_cell[1]] != FREE:
            
            path = None
            continue

        goal_row, goal_col = next_cell
        new_x = goal_col * resolution + resolution / 2
        new_y = goal_row * resolution + resolution / 2

        total_distance += resolution  
        robot_state = (new_x, new_y, 0.0)
        scan_needed = True

        path = path[1:]
        if len(path) == 1:
            pending_check = target_cell
            path = None

    point_cloud = np.vstack(point_cloud_parts) if point_cloud_parts else np.empty((0, 2))
    stats = {
        "steps": steps,
        "distance": total_distance,
        "cells_known": int(np.sum(global_map.grid != 0)),
        "coverage_history": coverage_history,
        "trajectory": trajectory,
    }
    return global_map, point_cloud, stats
