import numpy as np
from scipy.spatial import KDTree

def old_create_boundary(cell_points: np.ndarray, num_cells: int, num_comparison_points: int) -> np.ndarray:
    kd_tree = KDTree(cell_points)
    x_min, y_min = np.min(cell_points, axis=0)
    x_max, y_max = np.max(cell_points, axis=0)

    top_edge = np.linspace([x_min, y_max], [x_max, y_max], num_comparison_points)
    right_edge = np.linspace([x_max, y_max], [x_max, y_min], num_comparison_points)
    bottom_edge = np.linspace([x_max, y_min], [x_min, y_min], num_comparison_points)
    left_edge = np.linspace([x_min, y_min], [x_min, y_max], num_comparison_points)

    boundary_cycle_mask = np.zeros(num_cells, dtype=bool)
    _, indices = kd_tree.query(np.concatenate([top_edge, right_edge, bottom_edge, left_edge], axis=0), k=1)
    boundary_cycle_mask[indices] = 1

    return boundary_cycle_mask


def create_boundary(triangles: np.ndarray, max_cells: int):
    valid_mask = ~np.any(triangles == -1, axis=1)
    valid_triangles = triangles[valid_mask]
    
    edge_1 = valid_triangles[:, [0, 1]]
    edge_2 = valid_triangles[:, [1, 2]]
    edge_3 = valid_triangles[:, [2, 0]]
    edges = np.concatenate([edge_1, edge_2, edge_3], axis=0)
    edges = np.sort(edges, axis=1)

    sorted_indices = np.lexsort((edges[:, 1], edges[:, 0]))
    sorted_edges = edges[sorted_indices]

    equal_next = np.all(sorted_edges[1:] == sorted_edges[:-1], axis=1)
    is_unique = np.concatenate([np.array([True]), ~equal_next]) & np.concatenate([~equal_next, np.array([True])])

    boundary_edges = sorted_edges[is_unique]
    boundary_indices = np.unique(boundary_edges.flatten())
   
    boundary_cycle_mask = np.zeros(max_cells, dtype=bool)
    boundary_cycle_mask[boundary_indices] = 1
    
    return boundary_cycle_mask

def constrain_to_cycle(adjacency: np.ndarray, boundary_cycle_mask: np.ndarray, cell_points: np.ndarray, max_cells: int, max_degree: int):
    boundary_indices = np.where(boundary_cycle_mask)[0]
    boundary_points = cell_points[boundary_indices]
    center = np.mean(boundary_points, axis=0)
    relative_points = boundary_points - center
    angles = np.arctan2(relative_points[:, 1], relative_points[:, 0])
    sorted_indices = np.argsort(angles)
    sorted_boundary_indices = boundary_indices[sorted_indices]

    boundary_len = np.sum(boundary_cycle_mask)
    for i in range(boundary_len):
        current = sorted_boundary_indices[i]
        before = sorted_boundary_indices[(i - 1) % boundary_len]
        after = sorted_boundary_indices[(i + 1) % boundary_len]
        
        mask = adjacency[current] != -1
        neighbours = adjacency[current][mask]

        boundary_mask = np.zeros(max_cells, dtype=bool)
        boundary_mask[sorted_boundary_indices] = True
        non_boundary_mask = ~boundary_mask[neighbours]
        non_boundary_neighbours = neighbours[non_boundary_mask]
        num_non_boundary = non_boundary_neighbours.shape[0]

        new_connectivity = np.full(max_degree, -1, dtype=np.int32)
        new_connectivity[:num_non_boundary] = non_boundary_neighbours
        new_connectivity[num_non_boundary] = before
        new_connectivity[num_non_boundary+1] = after

        adjacency[current] = new_connectivity


def remove_excessive_boundary_cells(adjacency: np.ndarray, boundary_cycle_mask: np.ndarray, cell_points: np.ndarray, cell_types: np.ndarray, target_areas: np.ndarray):
    boundary_indices = np.where(boundary_cycle_mask)[0]
    boundary_points = cell_points[boundary_indices]
    center = np.mean(boundary_points, axis=0)
    relative_points = boundary_points - center
    angles = np.arctan2(relative_points[:, 1], relative_points[:, 0])
    sorted_indices = np.argsort(angles)
    sorted_boundary_indices = boundary_indices[sorted_indices]
   
    boundary_len = np.sum(boundary_cycle_mask)
    for i in range(boundary_len):
        current = sorted_boundary_indices[i]
        before = sorted_boundary_indices[(i - 1) % boundary_len]
        after = sorted_boundary_indices[(i + 1) % boundary_len]

        neighbours = adjacency[current]
        boundary_neighbours = np.intersect1d(neighbours, sorted_boundary_indices)
        remove_flag = False
        if len(boundary_neighbours) <= 2:
            bc = cell_points[before] - cell_points[current]
            ac = cell_points[after] - cell_points[current]
            pair_dot = np.dot(bc, ac)
            angle = np.arccos(pair_dot / (np.linalg.norm(bc) * np.linalg.norm(ac)))

            if angle < np.pi / 2:
                remove_flag = True
         
    #    print(boundary_neighbours, neighbours[neighbours != -1])
        if len(boundary_neighbours) == len(neighbours[neighbours != -1]) and (len(boundary_neighbours) + len(neighbours[neighbours != -1]) > 0):
            remove_flag = True
                 
        if remove_flag:
            cell_points[current] = [-1, -1]
            boundary_cycle_mask[current] = 0
            target_areas[current] = 0

    return np.sum(~(cell_points == -1).all(axis=1))


def add_additional_boundary_cells(adjacency: np.ndarray, boundary_cycle_mask: np.ndarray, cell_points: np.ndarray, cell_types: np.ndarray, target_areas: np.ndarray):
    boundary_indices = np.where(boundary_cycle_mask)[0]
    boundary_points = cell_points[boundary_indices]
    center = np.mean(boundary_points, axis=0)
    relative_points = boundary_points - center
    angles = np.arctan2(relative_points[:, 1], relative_points[:, 0])
    sorted_indices = np.argsort(angles)
    sorted_boundary_indices = boundary_indices[sorted_indices]

    edges = np.stack((sorted_boundary_indices, np.roll(sorted_boundary_indices, shift=-1)), axis=1)
    for i in range(len(edges)):
        a_index, b_index = edges[i]
        shared_cells = np.intersect1d(adjacency[a_index], adjacency[b_index])
        shared_cells = shared_cells[shared_cells != -1]
        shared_non_boundary = np.setdiff1d(shared_cells, sorted_boundary_indices)

        if len(shared_non_boundary) == 0:
            continue

        c = cell_points[shared_non_boundary[0]]
        a = cell_points[a_index]
        b = cell_points[b_index]

        ac = c - a
        bc = c - b
        ac_norm = np.linalg.norm(ac)
        bc_norm = np.linalg.norm(bc)
        ac_dot_bc = np.dot(ac, bc)

        angle_cos = ac_dot_bc / (ac_norm * bc_norm)
        if angle_cos < 0:
            edge_vector = b - a
            edge_norm = np.linalg.norm(edge_vector)
            edge_unit_vector = edge_vector / edge_norm

            projection_length = np.dot(ac, edge_unit_vector)
            projection_vector = projection_length * edge_unit_vector

            reflected_point = 2 * (a + projection_vector) - c
            
            vacancy = np.argwhere(cell_points == -1)[0][0]
            cell_points[vacancy] = reflected_point
            cell_types[vacancy] = 1
            target_areas[vacancy] = 0
            boundary_cycle_mask[vacancy] = True

    return np.sum(~(cell_points == -1).all(axis=1))
