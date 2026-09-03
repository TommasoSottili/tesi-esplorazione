
import numpy as np
from matplotlib.colors import ListedColormap


UNKNOWN = 0
FREE = 1
OCCUPIED = 2
INFLATED = 3   # cella libera ma troppo vicina a un ostacolo: vietata alla navigazione

class GlobalMap:
    def __init__(self, world_width, world_height, resolution=0.2):
       
        self.world_width = world_width
        self.world_height = world_height
        self.resolution = resolution

        # numero di celle per lato 
        self.n_rows = int(world_height / resolution)
        self.n_cols = int(world_width / resolution)

        # griglia globale inizialmente tutta sconosciuta
        self.grid = np.full((self.n_rows, self.n_cols), UNKNOWN, dtype=int)

    def world_to_cell(self, wx, wy):
        col = int(wx / self.resolution)
        row = int(wy / self.resolution)

        if row < 0 or row >= self.n_rows or col < 0 or col >= self.n_cols:
            return None
        return (row, col)

    def update(self, local_grid, robot_state, window_size=4.0):
        
        robot_x, robot_y, _ = robot_state
        half = window_size / 2.0
        n_local = local_grid.shape[0]   # celle per lato della griglia locale 

        # scorro tutte le celle della griglia locale
        for local_row in range(n_local):
            for local_col in range(n_local):
                state = local_grid[local_row][local_col]

                # se la cella locale è sconosciuta non porta informazione
                if state == UNKNOWN:
                    continue

                # converto la cella locale in coordinate del mondo
                wx = robot_x - half + (local_col + 0.5) * self.resolution
                wy = robot_y - half + (local_row + 0.5) * self.resolution

                # trovo la cella globale corrispondente
                cell = self.world_to_cell(wx, wy)
                if cell is None:
                    continue   # fuori dai confini del mondo, salto

                global_row, global_col = cell
                current = self.grid[global_row][global_col]

                if current == UNKNOWN:
                    # non sapevo nulla, prendo quello che dice la locale
                    self.grid[global_row][global_col] = state
                elif state == OCCUPIED:
                    # l'occupato ha priorità, lo scrivo sempre
                    self.grid[global_row][global_col] = OCCUPIED
                # altrimenti lascio com'è
    
    def plot(self, ax):
        from matplotlib.colors import ListedColormap

        # l'ordine dei colori corrisponde ai valori 0,1,2,3
        colormap = ListedColormap(['gray', 'white', 'black', 'red'])
        ax.imshow(self.grid, cmap=colormap, origin='lower',
                  vmin=0, vmax=3,
                  extent=[0, self.world_width, 0, self.world_height])
        ax.set_title("Mappa globale (grigio=ignoto, bianco=libero, nero=occupato, rosso=sicurezza)")
    
    def inflate_obstacles(self, inflation_radius=0.4):
        
        # raggio di inflation espresso in numero di celle
        radius_cells = int(inflation_radius / self.resolution)

        # trovo le coordinate (riga, colonna) di tutte le celle occupate
        occupied_cells = np.argwhere(self.grid == OCCUPIED)

        # per ogni cella occupata, gonfio attorno
        for (row, col) in occupied_cells:
            # scorro il quadrato di celle attorno a quella occupata
            for dr in range(-radius_cells, radius_cells + 1):
                for dc in range(-radius_cells, radius_cells + 1):
                    r = row + dr
                    c = col + dc

                    # salto se fuori dai confini della griglia
                    if r < 0 or r >= self.n_rows or c < 0 or c >= self.n_cols:
                        continue

                    # controllo che la cella sia davvero entro il raggio (cerchio, non quadrato)
                    if dr * dr + dc * dc > radius_cells * radius_cells:
                        continue

                    # gonfio SOLO le celle libere (non tocco occupate, sconosciute, già inflated)
                    if self.grid[r][c] == FREE:
                        self.grid[r][c] = INFLATED       