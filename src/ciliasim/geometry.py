import numpy as np
from scipy.stats import qmc

def uniform_random_layout(x: int, y: int, num_cells: int, max_cells: int) -> np.ndarray:
    radius = 1 / np.sqrt(num_cells)
    points = []

    while len(points) != num_cells:
        radius = radius * 0.95
        pd_sampler = qmc.PoissonDisk(d=2, radius=radius)
        points = pd_sampler.random(n=num_cells) * [x - 1, y - 1]
        points += 0.5

    full_points = np.full((max_cells, 2), -1, dtype=np.float32)
    full_points[:len(points)] = np.array(points)

    return full_points 


def hexagonal_layout(x: int, y: int, num_cells: int, max_cells: int) -> np.ndarray:
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

    full_points = np.full((max_cells, 2), -1, dtype=np.float32)
    full_points[:len(points)] = np.array(points)

    return full_points 


def uniform_target_areas(cell_types: np.ndarray, target_area: float) -> np.ndarray:
    target_areas = np.full_like(cell_types, target_area)
    target_areas[np.where(cell_types == 1)[0]] = 0
    target_areas[np.where(cell_types == -1)[0]] = -1

    return target_areas


def normal_random_target_areas(cell_types: np.ndarray, num_cells: int, mean: float, std: float) -> np.ndarray:
    target_areas = np.random.normal(loc=mean, scale=std, size=num_cells)
    target_areas[np.where(cell_types == 1)[0]] = 0
    target_areas[np.where(cell_types == -1)[0]] = -1

    return target_areas


def calculate_circumcenters(cell_points: np.ndarray, triangles: np.ndarray):
    mask = np.all(triangles != -1, axis=1)
    valid_triangles = triangles[mask]

    A = cell_points[valid_triangles[:, 0]]
    B = cell_points[valid_triangles[:, 1]]
    C = cell_points[valid_triangles[:, 2]]

    a2 = np.sum(A**2, axis=1)
    b2 = np.sum(B**2, axis=1)
    c2 = np.sum(C**2, axis=1)

    ax, ay = A[:, 0], A[:, 1]
    bx, by = B[:, 0], B[:, 1]
    cx, cy = C[:, 0], C[:, 1]

    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

    edge1 = B - A
    edge2 = C - B
    edge3 = A - C
    max_edge = np.maximum.reduce([
        np.linalg.norm(edge1, axis=1),
        np.linalg.norm(edge2, axis=1),
        np.linalg.norm(edge3, axis=1)
    ])

    eps = 1e-8 * np.maximum(1.0, max_edge ** 2)
    degenerate = np.abs(d) < eps

    ux_num = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by))
    uy_num = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax))

    ux = np.empty_like(ux_num, dtype=np.float64)
    uy = np.empty_like(uy_num, dtype=np.float64)

    nondeg_idx = ~degenerate
    if np.any(nondeg_idx):
        ux[nondeg_idx] = ux_num[nondeg_idx] / d[nondeg_idx]
        uy[nondeg_idx] = uy_num[nondeg_idx] / d[nondeg_idx]

    if np.any(degenerate):
        centers_deg = (A[degenerate] + B[degenerate] + C[degenerate]) / 3.0
        ux[degenerate] = centers_deg[:, 0]
        uy[degenerate] = centers_deg[:, 1]

    centers = np.stack([ux, uy], axis=1).astype(np.float32)

    full_centers = np.full((triangles.shape[0], 2), -1.0, dtype=np.float32)
    full_centers[mask] = centers

    return full_centers

def calculate_cell_area(cell_index: int, cell_point: np.ndarray, triangles: np.ndarray, circumcenters: np.ndarray):
    mask = np.any(triangles == cell_index, axis=1)
    vertices = circumcenters[mask]
    relative = vertices - cell_point
    angles = np.arctan2(relative[:, 1], relative[:, 0])
    sorted_indices = np.argsort(angles)
    sorted_vertices = vertices[sorted_indices]

    x = sorted_vertices[:, 0]
    y = sorted_vertices[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area

