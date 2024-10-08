import numpy as np
from numba import njit, jit
from numba.typed import List

@njit(cache=True)
def polygon_area(vertices: np.ndarray):
    x = vertices[:, 0]
    y = vertices[:, 1]

    sum_a = np.dot(x, y[np.arange(len(y)) - 1])
    sum_b = np.dot(y, x[np.arange(len(x)) - 1])

    return 0.5 * np.abs(sum_a - sum_b)

@njit(cache=True)
def polygon_perimeter(vertices: np.ndarray):
    looped_vertices = np.append(vertices, [vertices[0]], axis=0)
    differences = np.diff(looped_vertices, axis=0)
    distances = np.sqrt((differences ** 2).sum(axis=1))

    return np.sum(distances)

@njit(cache=True)
def calculate_force_matrix(
        num_cells: int,
        target_spring_length: float,
        critical_length_delta: float,
        cell_points: np.ndarray,
        cell_types: np.ndarray,
        target_areas: np.ndarray,
        voronoi_vertices: list, 
        adjacency_matrix: np.ndarray
    ):
    spring_matrix = np.zeros((num_cells, num_cells), dtype=np.float64)
    pressure_matrix = np.zeros((num_cells, num_cells), dtype=np.float64)
    distance_matrix = np.zeros((num_cells, num_cells), dtype=np.float64)
    unit_vector_matrix = np.zeros((num_cells, num_cells, 2), dtype=np.float64)

    for i in range(num_cells):
        # Calculate spring forces
        neighbours = np.nonzero(adjacency_matrix[i])[0]
        neighbour_positions = cell_points[neighbours]
        differences = neighbour_positions - cell_points[i]
        #distances = np.linalg.norm(differences, axis=1)
        # FIXME: Numba doesn't like the axis argument for some reason
        distances = np.array([np.linalg.norm(differences[j]) for j in range(len(differences))])
        unit_vectors = differences / distances[:, np.newaxis]

        distance_matrix[i, neighbours] = distances
        unit_vector_matrix[i, neighbours] = unit_vectors

        spring_matrix[i, neighbours] += target_spring_length - distances
        if cell_types[i] != 1:
            area = polygon_area(voronoi_vertices[i])
            split_area_difference = (target_areas[i] - area) / len(neighbours)

            pressure_matrix[i, neighbours] += split_area_difference
            pressure_matrix[neighbours, i] += split_area_difference 

    spring_matrix[cell_types == 1] /= 10
    spring_matrix = np.clip(spring_matrix, -critical_length_delta, None)

    force_matrix = (spring_matrix + pressure_matrix).T[:, :, np.newaxis] * unit_vector_matrix

    return force_matrix, distance_matrix

@njit(cache=True)
def calculate_boundary_reflection(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    ac = c - a
    bc = c - b
    ac_dot_bc = np.dot(ac, bc)
    angle = np.arccos(ac_dot_bc / (np.linalg.norm(ac) * np.linalg.norm(bc)))

    if angle > np.pi / 2:
        edge_vector = b - a
        edge_unit_vector = edge_vector / np.linalg.norm(edge_vector)

        projection_length = np.dot(c - a, edge_unit_vector)
        projection_vector = projection_length * edge_unit_vector

        reflected_point = 2 * (a + projection_vector) - c
        return True, reflected_point

    return False, None

# FIXME: Actually might be so simple that compilation makes it worse
@njit(cache=True)
def hexagonal_grid_layout(num_cells: int, x: int, y: int):
    num_rings = int(np.floor(1/2 + np.sqrt(12 * num_cells - 3) / 6))

    points = []
    cx = x / 2
    cy = y / 2

    points.append((cx, cy))
    
    for i in range(1, num_rings + 1):
        for j in range(6 * i):
            angle = j * np.pi / (3 * i)
            if i % 2 == 0:
                angle += np.pi / (3 * i)
            x = cx + i * np.cos(angle)
            y = cy + i * np.sin(angle)
            points.append((x, y))

    return points
