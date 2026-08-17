# definizione del ground truth, contenente gli ostacoli del bosco e i confini dell' area

from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union, nearest_points

class Environment: 

  def __init__(self, xmin, ymin, xmax, ymax):

    self.obstacles = [] # lista degli ostacoi del bosco
    self.bounds = box(xmin, ymin, xmax, ymax) # confini rettangolari dell' area 
    self._union_cache = None
    self._union_dirty = True

  def add_circle(self, cx, cy, radius): # aggiunge un albero alla lista degli ostacoli

    albero = Point(cx, cy).buffer(radius)
    self.obstacles.append(albero)
    self._union_dirty = True

  def add_polygon(self, vertices): # aggiunge una roccia alla lista degli ostacoli

    roccia = Polygon(vertices)
    self.obstacles.append(roccia)
    self._union_dirty = True

  def _obstacles_union(self):

    if not self.obstacles:
      return None
    if self._union_dirty:
      self._union_cache = unary_union(self.obstacles)
      self._union_dirty = False
    return self._union_cache

  def first_intersection_with_line(self, line):

    union = self._obstacles_union()
    if union is None:
      return None
    intersezione = line.intersection(union)
    if intersezione.is_empty:
      return None
    origine = Point(line.coords[0])
    punto_piu_vicino = nearest_points(origine, intersezione)[1]
    return (punto_piu_vicino.x, punto_piu_vicino.y)

    
     
