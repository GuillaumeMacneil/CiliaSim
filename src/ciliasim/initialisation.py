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


def hexagonal_layout(x: int, y: int, num_cells: int, max_cells: int):
    num_rings = max(0, int(np.floor((-3 + np.sqrt(12 * num_cells - 3)) / 6)))
    actual_num_cells = 1 + 3 * num_rings * (num_rings + 1)

    if actual_num_cells > max_cells:
        raise ValueError(f"Hexagonal layout requires {actual_num_cells} cells, but max_cells={max_cells}.")

    cx = x / 2
    cy = y / 2

    points = [(cx, cy)]

    for ring in range(1, num_rings + 1):
        for j in range(6 * ring):
            angle = j * np.pi / (3 * ring)
            px = cx + ring * np.cos(angle)
            py = cy + ring * np.sin(angle)
            points.append((px, py))

    points = np.asarray(points, dtype=np.float32)

    full_points = np.full((max_cells, 2), -1, dtype=np.float32)
    full_points[:actual_num_cells] = points

    return full_points, actual_num_cells


def uniform_target_areas(cell_types: np.ndarray, target_area: float) -> np.ndarray:
    target_areas = np.full(cell_types.shape, target_area, dtype=np.float32)
    target_areas[np.where(cell_types == 1)[0]] = 0
    target_areas[np.where(cell_types == -1)[0]] = -1

    return target_areas


def normal_random_target_areas(cell_types: np.ndarray, num_cells: int, mean: float, std: float) -> np.ndarray:
    target_areas = np.random.normal(loc=mean, scale=std, size=num_cells)
    target_areas[np.where(cell_types == 1)[0]] = 0
    target_areas[np.where(cell_types == -1)[0]] = -1

    return target_areas
