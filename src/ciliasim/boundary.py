import numpy as np
from scipy.spatial import KDTree

# FIXME: I think this can be replaced with an exact method using triangulation
def create_boundary(cell_points: np.ndarray, num_cells: int, num_comparison_points: int) -> np.ndarray:
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
     
def remove_excessive_boundary_cells(adjacency: np.ndarray, boundary_cycle_mask: np.ndarray, cell_points: np.ndarray, cell_types: np.ndarray, target_areas: np.ndarray, num_cells: int):
    pass

def add_additional_boundary_cells(adjacency: np.ndarray, boundary_cycle_mask: np.ndarray, cell_points: np.ndarray, cell_types: np.ndarray, target_areas: np.ndarray, num_cells: int):
    pass
