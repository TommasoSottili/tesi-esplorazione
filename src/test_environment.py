
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from environment import Environment

# Creo un bosco di 20x20 metri
bosco = Environment(0, 0, 20, 20)

# Aggiungo qualche albero e una roccia
bosco.add_circle(5, 5, 0.8)    
bosco.add_circle(12, 8, 1.2)     
bosco.add_circle(8, 14, 0.5)     
bosco.add_polygon([(14, 14), (16, 15), (15, 17), (13, 16)]) 

# Preparo il foglio da disegno e ci disegno il bosco
fig, ax = plt.subplots(figsize=(7, 7))
bosco.plot(ax)

# Provo un raggio: parte da (2,2) e va verso (12,8)
raggio = LineString([(2, 2), (12, 8)])
impatto = bosco.first_intersection_with_line(raggio)

# Disegno il raggio e il punto d'impatto (pallino rosso)
ax.plot([2, 12], [2, 8], linestyle='--', color='blue')  
if impatto is not None:
    ax.plot(impatto[0], impatto[1], 'ro', markersize=10)
    print(f"Il raggio colpisce in: {impatto}")
else:
    print("Il raggio non colpisce nessun ostacolo")

plt.show()