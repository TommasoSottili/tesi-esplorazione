
import numpy as np
from matplotlib.colors import ListedColormap


UNKNOWN = 0
FREE = 1
OCCUPIED = 2
INFLATED = 3   

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
    
        col = int(wx / self.resolution + 1e-9)
        row = int(wy / self.resolution + 1e-9)

        if row < 0 or row >= self.n_rows or col < 0 or col >= self.n_cols:
            return None
        return (row, col)

    def update(self, local_grid, robot_state, window_size=4.0):

        robot_x, robot_y, _ = robot_state
        n_local = local_grid.shape[0]   # celle per lato della griglia locale

        robot_cell = self.world_to_cell(robot_x, robot_y)
        if robot_cell is None:
            return  

        robot_row, robot_col = robot_cell
        half_cells = n_local // 2  

        # scorro tutte le celle della griglia locale
        for local_row in range(n_local):
            for local_col in range(n_local):
                state = local_grid[local_row][local_col]

                # se la cella locale è sconosciuta non porta informazione
                if state == UNKNOWN:
                    continue

                global_row = robot_row - half_cells + local_row
                global_col = robot_col - half_cells + local_col

                if global_row < 0 or global_row >= self.n_rows or global_col < 0 or global_col >= self.n_cols:
                    continue

                current = self.grid[global_row][global_col]

                if current == UNKNOWN:
                    self.grid[global_row][global_col] = state
                elif state == OCCUPIED:
                    self.grid[global_row][global_col] = OCCUPIED
    
    def plot(self, ax):
        from matplotlib.colors import ListedColormap

        # l'ordine dei colori corrisponde ai valori 0,1,2,3
        colormap = ListedColormap(['gray', 'white', 'black', 'red'])
        ax.imshow(self.grid, cmap=colormap, origin='lower',
                  vmin=0, vmax=3,
                  extent=[0, self.world_width, 0, self.world_height])
        ax.set_title("Mappa globale (grigio=ignoto, bianco=libero, nero=occupato, rosso=sicurezza)")
    
    def inflate_obstacles(self, inflation_radius=0.2, robot_cell=None):

        # raggio di inflation espresso in numero di celle
        radius_cells = int(inflation_radius / self.resolution)

        # trovo le coordinate (riga, colonna) di tutte le celle occupate
        occupied_cells = np.argwhere(self.grid == OCCUPIED)

        for (row, col) in occupied_cells:
            # scorro il quadrato di celle attorno a quella occupata
            for dr in range(-radius_cells, radius_cells + 1):
                for dc in range(-radius_cells, radius_cells + 1):
                    r = row + dr
                    c = col + dc

                    # salto se fuori dai confini della griglia
                    if r < 0 or r >= self.n_rows or c < 0 or c >= self.n_cols:
                        continue

                    # controllo che la cella sia davvero entro il raggio 
                    if dr * dr + dc * dc > radius_cells * radius_cells:
                        continue

                    if robot_cell is not None and (r, c) == tuple(robot_cell):
                        continue

                    if self.grid[r][c] == FREE:
                        self.grid[r][c] = INFLATED

    def find_frontiers(self):
       
        frontiers = []

        # le 4 direzioni per guardare i vicini
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # scorro tutte le celle della griglia
        for row in range(self.n_rows):
            for col in range(self.n_cols):

                if self.grid[row][col] != FREE:
                    continue

                for dr, dc in directions:
                    r = row + dr
                    c = col + dc

                    # salto i vicini fuori dai confini
                    if r < 0 or r >= self.n_rows or c < 0 or c >= self.n_cols:
                        continue

                    if self.grid[r][c] == UNKNOWN:
                        frontiers.append((row, col))
                        break  

        return frontiers  