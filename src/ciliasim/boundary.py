import numpy as np
from scipy.spatial import ConvexHull

def create_boundary(cell_points: np.ndarray, num_cells: int, max_cells: int, spring_length: float):
    real_points = cell_points[:num_cells]

    hull = ConvexHull(real_points)
    hull_points = real_points[hull.vertices]

    center = np.mean(hull_points, axis=0)
    offset = spring_length
    spacing = spring_length

    relative_points = hull_points - center
    radial_distances = np.linalg.norm(relative_points, axis=1, keepdims=True)

    if np.any(radial_distances == 0):
        raise RuntimeError("Cannot construct boundary from degenerate convex hull.")

    offset_hull = hull_points + offset * relative_points / radial_distances

    ghost_points = []

    for i in range(len(offset_hull)):
        a = offset_hull[i]
        b = offset_hull[(i + 1) % len(offset_hull)]

        edge = b - a
        edge_length = np.linalg.norm(edge)

        if edge_length == 0:
            continue

        num_segments = max(1, int(np.ceil(edge_length / spacing)))

        for j in range(num_segments):
            ghost_points.append(a + (j / num_segments) * edge)

    ghost_points = np.asarray(ghost_points, dtype=cell_points.dtype)
    num_ghosts = len(ghost_points)

    if num_cells + num_ghosts > max_cells:
        raise RuntimeError(f"Boundary requires {num_ghosts} ghost cells, but only {max_cells - num_cells} slots are available.")

    boundary_cycle = np.arange(num_cells, num_cells + num_ghosts, dtype=np.int32)

    boundary_cycle_mask = np.zeros(max_cells, dtype=bool)
    boundary_cycle_mask[boundary_cycle] = True

    cell_points[boundary_cycle] = ghost_points

    return boundary_cycle_mask, boundary_cycle, num_cells + num_ghosts


def create_hexagonal_boundary(cell_points: np.ndarray, num_cells: int, max_cells: int, spring_length: float):
    num_rings = int(round((-3 + np.sqrt(12 * num_cells - 3)) / 6))

    if 1 + 3 * num_rings * (num_rings + 1) != num_cells:
        raise ValueError(f"num_cells={num_cells} is not a complete hexagonal lattice.")

    boundary_ring = num_rings + 1
    axial_points = []

    for q in range(-boundary_ring, boundary_ring + 1):
        for r in range(-boundary_ring, boundary_ring + 1):
            if max(abs(q), abs(r), abs(q + r)) == boundary_ring:
                axial_points.append((q, r))

    axial_points = np.asarray(axial_points, dtype=np.float32)

    q = axial_points[:, 0]
    r = axial_points[:, 1]

    real_center = np.mean(cell_points[:num_cells], axis=0)

    ghost_points = np.column_stack((
        real_center[0] + spring_length * (q + 0.5 * r),
        real_center[1] + spring_length * (np.sqrt(3) / 2) * r,
    )).astype(cell_points.dtype)

    relative_points = ghost_points - real_center
    angles = np.arctan2(relative_points[:, 1], relative_points[:, 0])
    order = np.argsort(angles)
    ghost_points = ghost_points[order]

    num_ghosts = len(ghost_points)

    if num_cells + num_ghosts > max_cells:
        raise RuntimeError(f"Boundary requires {num_ghosts} ghost cells, but only {max_cells - num_cells} slots are available.")

    boundary_cycle = np.arange(num_cells, num_cells + num_ghosts, dtype=np.int32)
    boundary_cycle_mask = np.zeros(max_cells, dtype=bool)
    boundary_cycle_mask[boundary_cycle] = True

    cell_points[boundary_cycle] = ghost_points

    return boundary_cycle_mask, boundary_cycle, num_cells + num_ghosts


def constrain_to_cycle(adjacency: np.ndarray, boundary_cycle: np.ndarray, boundary_cycle_mask: np.ndarray, max_degree: int):
    boundary_len = len(boundary_cycle)

    if boundary_len < 3:
        raise RuntimeError("Boundary must contain at least three cells.")

    for i, current in enumerate(boundary_cycle):
        before = boundary_cycle[i - 1]
        after = boundary_cycle[(i + 1) % boundary_len]

        neighbours = adjacency[current]
        neighbours = neighbours[neighbours >= 0]
        non_boundary_neighbours = neighbours[~boundary_cycle_mask[neighbours]]

        required_degree = len(non_boundary_neighbours) + 2

        if required_degree > max_degree:
            raise RuntimeError(f"Boundary cell {current} requires degree {required_degree}, but max_degree is {max_degree}.")

        new_connectivity = np.full(max_degree, -1, dtype=np.int32)
        new_connectivity[:len(non_boundary_neighbours)] = non_boundary_neighbours
        new_connectivity[len(non_boundary_neighbours)] = before
        new_connectivity[len(non_boundary_neighbours) + 1] = after
        adjacency[current] = new_connectivity


def add_additional_boundary_cells(adjacency: np.ndarray, boundary_cycle: np.ndarray, boundary_cycle_mask: np.ndarray, cell_points: np.ndarray, cell_types: np.ndarray, target_areas: np.ndarray):
    additions = []

    for i, a_index in enumerate(boundary_cycle):
        b_index = boundary_cycle[(i + 1) % len(boundary_cycle)]

        shared_cells = np.intersect1d(adjacency[a_index], adjacency[b_index])
        shared_cells = shared_cells[shared_cells >= 0]
        shared_non_boundary = shared_cells[~boundary_cycle_mask[shared_cells]]

        if len(shared_non_boundary) == 0:
            continue

        c_index = shared_non_boundary[0]

        a = cell_points[a_index]
        b = cell_points[b_index]
        c = cell_points[c_index]

        ac = c - a
        bc = c - b
        ac_norm = np.linalg.norm(ac)
        bc_norm = np.linalg.norm(bc)

        if ac_norm == 0 or bc_norm == 0:
            continue

        angle_cos = np.clip(np.dot(ac, bc) / (ac_norm * bc_norm), -1.0, 1.0)

        if angle_cos >= 0.0:
            continue

        edge_vector = b - a
        edge_norm = np.linalg.norm(edge_vector)

        if edge_norm == 0:
            continue

        edge_unit_vector = edge_vector / edge_norm
        projection = a + np.dot(c - a, edge_unit_vector) * edge_unit_vector
        reflected_point = 2 * projection - c

        active_mask = ~np.all(cell_points == -1, axis=1)
        active_indices = np.flatnonzero(active_mask)
        distances = np.linalg.norm(cell_points[active_indices] - reflected_point, axis=1)

        if len(distances) > 0 and np.min(distances) < 1e-6:
            existing_index = active_indices[np.argmin(distances)]

            if boundary_cycle_mask[existing_index]:
                additions.append((i + 1, existing_index))

            continue

        vacancies = np.flatnonzero(~active_mask)

        if len(vacancies) == 0:
            raise RuntimeError("No vacant cell slots available for additional boundary cell.")

        vacancy = int(vacancies[0])
        cell_points[vacancy] = reflected_point
        cell_types[vacancy] = 1
        target_areas[vacancy] = 0
        boundary_cycle_mask[vacancy] = True
        additions.append((i + 1, vacancy))

    if additions:
        offset = 0
        for position, cell in additions:
            if cell in boundary_cycle:
                continue
            boundary_cycle = np.insert(boundary_cycle, position + offset, cell)
            offset += 1

    num_cells = np.sum(~np.all(cell_points == -1, axis=1))

    return num_cells, boundary_cycle


def remove_excessive_boundary_cells(adjacency: np.ndarray, boundary_cycle: np.ndarray, boundary_cycle_mask: np.ndarray, cell_points: np.ndarray, cell_types: np.ndarray, target_areas: np.ndarray):
    remove_cells = []

    for i, current in enumerate(boundary_cycle):
        if cell_types[current] != 1:
            continue

        before = boundary_cycle[i - 1]
        after = boundary_cycle[(i + 1) % len(boundary_cycle)]

        neighbours = adjacency[current]
        neighbours = neighbours[neighbours >= 0]
        boundary_neighbours = neighbours[boundary_cycle_mask[neighbours]]

        remove_flag = False

        if len(boundary_neighbours) <= 2:
            bc = cell_points[before] - cell_points[current]
            ac = cell_points[after] - cell_points[current]
            bc_norm = np.linalg.norm(bc)
            ac_norm = np.linalg.norm(ac)

            if bc_norm > 0 and ac_norm > 0:
                angle = np.arccos(np.clip(np.dot(bc, ac) / (bc_norm * ac_norm), -1.0, 1.0))

                if angle < np.pi / 2:
                    remove_flag = True

        if len(neighbours) > 0 and len(boundary_neighbours) == len(neighbours):
            remove_flag = True

        if remove_flag:
            remove_cells.append(int(current))

    if remove_cells:
        remove_set = set(remove_cells)
        boundary_cycle = np.asarray([cell for cell in boundary_cycle if int(cell) not in remove_set], dtype=np.int32)

        for current in remove_cells:
            cell_points[current] = [-1, -1]
            boundary_cycle_mask[current] = False
            target_areas[current] = -1
            cell_types[current] = -1

    num_cells = np.sum(~np.all(cell_points == -1, axis=1))

    return num_cells, boundary_cycle
