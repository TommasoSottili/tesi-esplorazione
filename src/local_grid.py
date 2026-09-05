
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
    # (+1e-9 evita che un punto esattamente sul bordo tra due celle finisca
    # nella cella sbagliata per rumore di virgola mobile, es. 40.999999999 -> 40 invece di 41)
    col = int((dx + half) / resolution + 1e-9)
    row = int((dy + half) / resolution + 1e-9)

    # Numero totale di celle per lato
    n_cells = int(window_size / resolution)

    # Se il punto cade fuori dalla finestra, restituisco None
    if row < 0 or row >= n_cells or col < 0 or col >= n_cells:
        return None

    return (row, col)

def build_local_grid(robot_state, scan_points, window_size=4.0, resolution=0.2, r_max=2.0, free_margin=None):

    # robot_state: (x, y, theta) posizione e orientamento del robot
    # scan_points: array Nx2 dei punti visti dal LiDAR (uscita di sensor.scan)

    robot_x, robot_y, _ = robot_state

    # margine percettivo: quanto ci si ferma PRIMA del punto di impatto quando
    # si marcano le celle libere lungo un raggio che ha colpito un ostacolo.
    # Senza margine, un lembo sottile di ostacolo che taglia di striscio la
    # cella dell'impatto può non intercettare nessun campione del raggio
    # (specie da lontano/di lato) e finire marcato FREE per errore
    if free_margin is None:
        free_margin = resolution

    # Creo la griglia, inizialmente tutta sconosciuta
    n_cells = int(window_size / resolution)
    grid = np.full((n_cells, n_cells), UNKNOWN, dtype=int)

    # Per ogni punto della scansione
    for point in scan_points:
        px, py = point[0], point[1]

        # distanza del punto dal robot (distanza euclidea)
        distance = np.hypot(px - robot_x, py - robot_y)
        if distance < 1e-9:
            continue

        is_hit = distance < r_max - 1e-6

        # per i raggi a vuoto (nessun impatto) cammino fino in fondo; per quelli
        # che hanno colpito un ostacolo mi fermo un margine prima del punto di
        # impatto, per non dichiarare FREE l'incertezza attorno al bordo
        free_distance = max(distance - free_margin, 0.0) if is_hit else distance

        dir_x = (px - robot_x) / distance
        dir_y = (py - robot_y) / distance

        # cammino lungo il raggio a piccoli passi, marcando FREE le celle attraversate
        n_samples = int(free_distance / (resolution / 4))  # un campione ogni quarto di cella
        for i in range(n_samples):
            t = i / n_samples          # frazione del percorso (da 0 a quasi 1)
            sample_x = robot_x + t * free_distance * dir_x
            sample_y = robot_y + t * free_distance * dir_y

            cell = world_to_cell(sample_x, sample_y, robot_x, robot_y, window_size, resolution)
            if cell is not None:
                row, col = cell
                grid[row][col] = FREE

        # se il raggio ha colpito un ostacolo (non è arrivato a r_max), marco OCCUPIED
        if is_hit:
            cell = world_to_cell(px, py, robot_x, robot_y, window_size, resolution)
            if cell is not None:
                row, col = cell
                grid[row][col] = OCCUPIED

    return grid