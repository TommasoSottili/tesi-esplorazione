
import heapq
from global_map import FREE


def get_walkable_neighbors(cell, global_map):

    row, col = cell
    neighbors = []

    # le 4 direzioni: su, giù, sinistra, destra
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

def find_path(global_map, start, goal):
    
    # 1. Strutture dati di Dijkstra
    distances = {start: 0}          # distanza minima nota da start a ogni cella
    predecessors = {}               # da quale cella siamo arrivati a una data cella
    visited = set()                 # celle già definitivamente processate

    # 2. La coda di priorità: contiene coppie (distanza, cella)
    priority_queue = [(0, start)]

    # 3. Ciclo principale: espansione a onde
    while priority_queue:
        # estraggo la cella con distanza minore tra quelle in attesa
        current_dist, current = heapq.heappop(priority_queue)

        # se l'ho già processata, la salto (può capitare di averla in coda più volte)
        if current in visited:
            continue
        visited.add(current)

        # se ho raggiunto la destinazione, ho finito
        if current == goal:
            return reconstruct_path(predecessors, start, goal)

        # esamino i vicini percorribili
        for neighbor in get_walkable_neighbors(current, global_map):
            new_dist = current_dist + 1   # ogni passo costa 1 (celle adiacenti)

            # se ho trovato un percorso più corto verso questo vicino, lo aggiorno
            if neighbor not in distances or new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                predecessors[neighbor] = current
                heapq.heappush(priority_queue, (new_dist, neighbor))

    # 4. Se svuoto la coda senza raggiungere goal, non esiste percorso
    return None

def reconstruct_path(predecessors, start, goal):
    
    path = [goal]
    current = goal

    # risalgo di predecessore in predecessore, da goal fino a start
    while current != start:
        current = predecessors[current]
        path.append(current)

    # il percorso è stato costruito al contrario (da goal a start): lo giro
    path.reverse()
    return path