
import random
from pathfinding import get_walkable_neighbors


def _reachable_cells(global_map, start):
    # BFS sulle celle percorribili (FREE) raggiungibili da start, 4-connesse.
    if start is None:
        return set()

    visited = {start}
    stack = [start]

    while stack:
        current = stack.pop()
        for neighbor in get_walkable_neighbors(current, global_map):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)

    return visited


def random_strategy(global_map, frontiers, robot_cell):

    if not frontiers:
        return None

    # scarto le frontiere non raggiungibili dal robot (es. isolate dalla
    # fascia di sicurezza attorno agli ostacoli): sceglierle sprecherebbe
    # passi in pathfinding destinati a fallire. Scarto anche la cella del
    # robot stessa: può ricomparire come frontiera se uno dei suoi vicini
    # resta sconosciuto, ma sceglierla come target non produce un vero
    # movimento (find_path restituirebbe un percorso di un solo passo)
    reachable = _reachable_cells(global_map, robot_cell)
    reachable_frontiers = [f for f in frontiers if f in reachable and f != robot_cell]

    if not reachable_frontiers:
        return None

    return random.choice(reachable_frontiers)