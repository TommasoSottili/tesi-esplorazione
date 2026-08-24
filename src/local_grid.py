
import numpy as np

UNKNOWN = 0
FREE = 1
OCCUPIED = 2


def world_to_cell(wx, wy, robot_x, robot_y, window_size, resolution):
    
    # Posizione del punto relativa al robot (il robot è il centro della finestra)
    dx = wx - robot_x
    dy = wy - robot_y

    # L'angolo in basso a sinistra della finestra sta a -window_size/2 dal robot
    half = window_size / 2.0

    # Converto la posizione relativa in indici di cella
    col = int((dx + half) / resolution)
    row = int((dy + half) / resolution)

    # Numero totale di celle per lato
    n_cells = int(window_size / resolution)

    # Se il punto cade fuori dalla finestra, restituisco None
    if row < 0 or row >= n_cells or col < 0 or col >= n_cells:
        return None

    return (row, col)

def build_local_grid(robot_state, scan_points, window_size=4.0, resolution=0.2, r_max=6.0):
    
    # robot_state: (x, y, theta) posizione e orientamento del robot
    # scan_points: array Nx2 dei punti visti dal LiDAR (uscita di sensor.scan)

    robot_x, robot_y, _ = robot_state

    # Creo la griglia, inizialmente tutta sconosciuta
    n_cells = int(window_size / resolution)
    grid = np.full((n_cells, n_cells), UNKNOWN, dtype=int)

    # Per ogni punto della scansione
    for point in scan_points:
        px, py = point[0], point[1]

        # distanza del punto dal robot (distanza euclidea)
        distance = np.hypot(px - robot_x, py - robot_y)

        # cammino lungo il raggio a piccoli passi, marcando FREE le celle attraversate
        n_samples = int(distance / (resolution / 2))  # un campione ogni mezza cella
        for i in range(n_samples):
            t = i / n_samples          # frazione del percorso (da 0 a quasi 1)
            sample_x = robot_x + t * (px - robot_x)
            sample_y = robot_y + t * (py - robot_y)

            cell = world_to_cell(sample_x, sample_y, robot_x, robot_y, window_size, resolution)
            if cell is not None:
                row, col = cell
                grid[row][col] = FREE

        # se il raggio ha colpito un ostacolo (non è arrivato a r_max), marco OCCUPIED
        if distance < r_max - 1e-6:
            cell = world_to_cell(px, py, robot_x, robot_y, window_size, resolution)
            if cell is not None:
                row, col = cell
                grid[row][col] = OCCUPIED

    return grid