import numpy as np
from scipy.spatial import Delaunay
from itertools import combinations


def full_update(num_cells: int, max_cells: int, max_degree: int, max_triangles: int, cell_points: np.ndarray):
    adjacency = np.full((max_cells, max_degree), -1, dtype=np.int32)
    triangles = np.full((max_triangles, 3), -1, dtype=np.int32)
    valid = np.all(cell_points != -1, axis=1)
    delaunay = Delaunay(cell_points[valid])

    triangle_pointer = 0 
    indices = np.zeros(num_cells, dtype=np.int8)
    ordered_simplices = np.sort(delaunay.simplices, axis=1)
    for simplex in ordered_simplices:
        i, j, k = simplex
        add_neighbour(indices, adjacency, i, j)
        add_neighbour(indices, adjacency, j, k)
        add_neighbour(indices, adjacency, k, i)
        triangles[triangle_pointer] = [i, j, k]     
        triangle_pointer += 1
        
    return adjacency, triangles


def add_neighbour(indices: np.ndarray, adjacency: np.ndarray, a: int, b: int):
    pointer_a = indices[a]
    pointer_b = indices[b]
    adjacency[a][pointer_a] = b
    adjacency[b][pointer_b] = a
    indices[a] += 1
    indices[b] += 1


# FIXME: This should properly implement a partial update
def partial_update(adjacency: np.ndarray, num_cells: int, max_degree: int, cell_points: np.ndarray, triangles: np.ndarray):
    pass

