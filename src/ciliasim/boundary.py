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


def remove_excessive_boundary_cells(adjacency: np.ndarray, boundary_cycle_mask: np.ndarray, cell_points: np.ndarray, cell_types: np.ndarray, target_areas: np.ndarray, num_cells: int):
    # Any point with 2 neighbours or fewer needs to be removed
    pass


def add_additional_boundary_cells(adjacency: np.ndarray, boundary_cycle_mask: np.ndarray, cell_points: np.ndarray, cell_types: np.ndarray, target_areas: np.ndarray, num_cells: int):
    boundary_indices = np.where(boundary_cycle_mask)[0]
    boundary_points = cell_points[boundary_indices]
    center = np.mean(boundary_points, axis=0)
    relative_points = boundary_points - center
    angles = np.arctan2(relative_points[:, 1], relative_points[:, 0])
    sorted_indices = np.argsort(angles)
    sorted_boundary_indices = boundary_indices[sorted_indices]

    edges = np.stack((sorted_boundary_indices, np.roll(sorted_boundary_indices, shift=-1)), axis=1)
    for i in range(len(edges)):
        pass
    
