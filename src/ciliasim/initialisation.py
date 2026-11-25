import numpy as np
from scipy.stats import qmc


def uniform_random_layout(x: int, y: int, num_cells: int, max_cells: int, spring_length: float) -> np.ndarray:
    radius = spring_length
    points = []

    while len(points) != num_cells:
        pd_sampler = qmc.PoissonDisk(d=2, radius=radius)
        points = pd_sampler.random(n=num_cells) * [x - 1, y - 1]
        points += 0.5

        if len(points) != num_cells:
            radius *= 0.95

    full_points = np.full((max_cells, 2), -1, dtype=np.float32)
    full_points[:num_cells] = np.asarray(points, dtype=np.float32)

    return full_points


def hexagonal_layout(x: int, y: int, num_cells: int, max_cells: int, spring_length: float):
    num_rings = max(0, int(np.floor((-3 + np.sqrt(12 * num_cells - 3)) / 6)))
    actual_num_cells = 1 + 3 * num_rings * (num_rings + 1)

    if actual_num_cells > max_cells:
        raise ValueError(f"Hexagonal layout requires {actual_num_cells} cells, but max_cells={max_cells}.")

    axial_points = []

    for q in range(-num_rings, num_rings + 1):
        r_min = max(-num_rings, -q - num_rings)
        r_max = min(num_rings, -q + num_rings)

        for r in range(r_min, r_max + 1):
            axial_points.append((q, r))

    axial_points = np.asarray(axial_points, dtype=np.float32)

    q = axial_points[:, 0]
    r = axial_points[:, 1]

    points = np.column_stack((
        x / 2 + spring_length * (q + 0.5 * r),
        y / 2 + spring_length * (np.sqrt(3) / 2) * r,
    )).astype(np.float32)

    full_points = np.full((max_cells, 2), -1, dtype=np.float32)
    full_points[:actual_num_cells] = points

    return full_points, actual_num_cells


def uniform_target_areas(cell_types: np.ndarray, target_area: float) -> np.ndarray:
    target_areas = np.full(cell_types.shape, target_area, dtype=np.float32)
    target_areas[cell_types == 1] = 0.0
    target_areas[cell_types == -1] = -1.0

    return target_areas


def normal_random_target_areas(cell_types: np.ndarray, mean: float, std: float) -> np.ndarray:
    target_areas = np.full(cell_types.shape, -1.0, dtype=np.float32)

    real_mask = (cell_types != 1) & (cell_types != -1)
    target_areas[real_mask] = np.random.normal(loc=mean, scale=std, size=np.sum(real_mask))
    target_areas[cell_types == 1] = 0.0

    return target_areas
