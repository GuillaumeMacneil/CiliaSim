import matplotlib
import numpy as np
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt

from ciliasim import geometry
from ciliasim import tissue

tissue = tissue.Tissue(
            x = 10, 
            y = 10, 
            density = 0.06,
        )

tissue.specify_cells((1.0, 0.0))
triangles = tissue.triangles
points = tissue.cell_points

circumcenters = geometry.calculate_circumcenters(tissue.cell_points, tissue.triangles)

plt.figure(figsize=(8, 8))

plt.scatter(points[:, 0], points[:, 1], color='blue', label='Points')

plt.scatter(circumcenters[:, 0], circumcenters[:, 1], color='red', marker='x', label='Circumcenters')

adjacency = tissue.adjacency
n = adjacency.shape[0]
for i in range(n):
    for j in adjacency[i]:
        if j == -1 or i > j:
            continue  # skip padding and avoid duplicate lines
        x_coords = [points[i, 0], points[j, 0]]
        y_coords = [points[i, 1], points[j, 1]]
        plt.plot(x_coords, y_coords, color='gray', alpha=0.5)

for tri in triangles:
    if -1 in tri:  # Skip invalid triangles
        continue
    polygon = points[tri]
    plt.plot(*zip(*np.append(polygon, [polygon[0]], axis=0)), color='green', alpha=0.5)

plt.show()
