import numpy as np
from scipy.spatial import Delaunay


def full_update(num_cells: int, max_cells: int, max_degree: int, max_triangles: int, cell_points: np.ndarray, boundary_cycle: np.ndarray | None = None, boundary_cycle_mask: np.ndarray | None = None):
    delaunay = Delaunay(cell_points[:num_cells])
    simplices = np.sort(delaunay.simplices, axis=1)

    if boundary_cycle is not None and boundary_cycle_mask is not None:
        simplices = filter_boundary_triangles(simplices, boundary_cycle, boundary_cycle_mask)

    return build_connectivity(simplices, num_cells, max_cells, max_degree, max_triangles, cell_points)


def filter_boundary_triangles(simplices: np.ndarray, boundary_cycle: np.ndarray, boundary_cycle_mask: np.ndarray):
    allowed_edges = set()

    for i, a in enumerate(boundary_cycle):
        b = boundary_cycle[(i + 1) % len(boundary_cycle)]
        allowed_edges.add((min(int(a), int(b)), max(int(a), int(b))))

    keep = np.ones(len(simplices), dtype=bool)

    for triangle_index, triangle in enumerate(simplices):
        boundary_vertices = triangle[boundary_cycle_mask[triangle]]

        if len(boundary_vertices) < 2:
            continue

        for i in range(len(boundary_vertices)):
            for j in range(i + 1, len(boundary_vertices)):
                a = int(boundary_vertices[i])
                b = int(boundary_vertices[j])
                edge = (min(a, b), max(a, b))

                if edge not in allowed_edges:
                    keep[triangle_index] = False
                    break

            if not keep[triangle_index]:
                break

    return simplices[keep]


def build_connectivity(simplices: np.ndarray, num_cells: int, max_cells: int, max_degree: int, max_triangles: int, cell_points: np.ndarray):
    adjacency = np.full((max_cells, max_degree), -1, dtype=np.int32)
    triangles = np.full((max_triangles, 3), -1, dtype=np.int32)
    cell_triangles = np.full((max_cells, max_degree), -1, dtype=np.int32)

    neighbour_counts = np.zeros(max_cells, dtype=np.int32)
    triangle_counts = np.zeros(max_cells, dtype=np.int32)

    if len(simplices) > max_triangles:
        raise RuntimeError(f"Triangulation requires {len(simplices)} triangles, but max_triangles is {max_triangles}.")

    for triangle_index, simplex in enumerate(simplices):
        i, j, k = simplex

        add_neighbour(neighbour_counts, adjacency, i, j)
        add_neighbour(neighbour_counts, adjacency, j, k)
        add_neighbour(neighbour_counts, adjacency, k, i)

        triangles[triangle_index] = simplex

        for cell in simplex:
            if triangle_counts[cell] >= max_degree:
                raise RuntimeError(f"Cell {cell} belongs to more than max_degree={max_degree} triangles.")

            cell_triangles[cell, triangle_counts[cell]] = triangle_index
            triangle_counts[cell] += 1

    for cell in range(num_cells):
        count = triangle_counts[cell]

        if count <= 1:
            continue

        triangle_indices = cell_triangles[cell, :count]
        cell_simplices = triangles[triangle_indices]
        centroids = np.mean(cell_points[cell_simplices], axis=1)
        relative = centroids - cell_points[cell]
        angles = np.arctan2(relative[:, 1], relative[:, 0])
        order = np.argsort(angles)
        cell_triangles[cell, :count] = triangle_indices[order]

    return adjacency, triangles, cell_triangles


def add_neighbour(indices: np.ndarray, adjacency: np.ndarray, a: int, b: int):
    if not np.any(adjacency[a, :indices[a]] == b):
        if indices[a] >= adjacency.shape[1]:
            raise RuntimeError(f"Cell {a} exceeds maximum degree {adjacency.shape[1]}.")

        adjacency[a, indices[a]] = b
        indices[a] += 1

    if not np.any(adjacency[b, :indices[b]] == a):
        if indices[b] >= adjacency.shape[1]:
            raise RuntimeError(f"Cell {b} exceeds maximum degree {adjacency.shape[1]}.")

        adjacency[b, indices[b]] = a
        indices[b] += 1

# TODO: This should properly implement partial re-triangulation
def partial_update(adjacency: np.ndarray, num_cells: int, max_degree: int, cell_points: np.ndarray, triangles: np.ndarray):
    pass
