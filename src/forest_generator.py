import random
import math
from shapely.geometry import Point
from environment import Environment
from shapely.geometry import Point, Polygon

def is_position_valid(geometry, environment, start_point, clearance):

  for existing_obstacle in environment.obstacles:
    if geometry.intersects(existing_obstacle):
      return False
    
  if geometry.distance(start_point) < clearance:
    return False

  if not environment.bounds.contains(geometry):
        return False

  return True

def generate_forest(width, height, n_trees, n_rocks=0, seed=None, start_point=None, clearance=1.5, max_attempts=100):
   
    random.seed(seed)

    environment = Environment(0, 0, width, height)

    if start_point is None:
        start_point = Point(width / 2, height / 2)

    for _ in range(n_trees):
        for _ in range(max_attempts):
            cx = random.uniform(0, width)
            cy = random.uniform(0, height)
            radius = random.uniform(0.3, 1.0)

            candidate = Point(cx, cy).buffer(radius)

            if is_position_valid(candidate, environment, start_point, clearance):
                environment.add_circle(cx, cy, radius)
                break

    for _ in range(n_rocks):
        for _ in range(max_attempts):
            cx = random.uniform(0, width)
            cy = random.uniform(0, height)
            size = random.uniform(0.4, 1.0)

            vertices = generate_random_polygon(cx, cy, size)
            candidate = Polygon(vertices)

            if is_position_valid(candidate, environment, start_point, clearance):
                environment.add_polygon(vertices)
                break

    return environment

def generate_random_polygon(cx, cy, size, n_vertices=6):
   
    vertices = []
    for i in range(n_vertices):
        angle = (2 * math.pi / n_vertices) * i
        r = size * random.uniform(0.6, 1.0)
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        vertices.append((x, y))
    return vertices