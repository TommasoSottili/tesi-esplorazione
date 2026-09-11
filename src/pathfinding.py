
import heapq
from collections import deque
from global_map import FREE


def get_walkable_neighbors(cell, global_map):

    row, col = cell
    neighbors = []

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in directions:
        r = row + dr
        c = col + dc

        # scarto se fuori dai confini della griglia
        if r < 0 or r >= global_map.n_rows or c < 0 or c >= global_map.n_cols:
            continue

        # accetto solo se la cella è libera (percorribile)
        if global_map.grid[r][c] == FREE:
            neighbors.append((r, c))

    return neighbors

def reachable_cells_with_distance(global_map, start):
    
    if start is None:
        return {}

    distances = {start: 0}
    queue = deque([start])

    while queue:
        current = queue.popleft()
        for neighbor in get_walkable_neighbors(current, global_map):
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)

    return distances

def reachable_cells(global_map, start):
    # qualunque strategia può usarla per
    # sapere quali celle sono davvero raggiungibili dal robot, senza
    # duplicare la logica di percorribilità già definita sopra.
    return reachable_cells_with_distance(global_map, start).keys()

def find_path(global_map, start, goal):
    
    distances = {start: 0}        
    predecessors = {}             
    visited = set()                

    # 2. La coda di priorità: contiene coppie (distanza, cella)
    priority_queue = [(0, start)]

    # 3. espansione a onde
    while priority_queue:
        # estraggo la cella con distanza minore tra quelle in attesa
        current_dist, current = heapq.heappop(priority_queue)

        if current in visited:
            continue
        visited.add(current)

        # se ho raggiunto la destinazione, ho finito
        if current == goal:
            return reconstruct_path(predecessors, start, goal)

        # esamino i vicini percorribili
        for neighbor in get_walkable_neighbors(current, global_map):
            new_dist = current_dist + 1  

            # se ho trovato un percorso più corto verso questo vicino, lo aggiorno
            if neighbor not in distances or new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                predecessors[neighbor] = current
                heapq.heappush(priority_queue, (new_dist, neighbor))

    return None

def reconstruct_path(predecessors, start, goal):
    
    path = [goal]
    current = goal

    while current != start:
        current = predecessors[current]
        path.append(current)

    path.reverse()
    return path