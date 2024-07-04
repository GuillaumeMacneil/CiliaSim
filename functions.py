import numpy as np
from scipy.spatial import Voronoi

def polygon_area(vertices: np.ndarray):
    """
    Calculates the area of a polygon given its vertices using the shoelace formula.

    Args:
        vertices (np.ndarray): The vertices of the polygon.

    Returns:
        float: The area of the polygon.
    """
    x = vertices[:, 0]
    y = vertices[:, 1]

    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def is_center(polygon_vertices: np.ndarray, candidates: np.ndarray) -> int:
    """
    Finds the index of the candidate point closest to the polygon vertices.

    Args:
        polygon_vertices (np.ndarray): A 2D array of shape (N, 2) representing the vertices of the polygon.
        candidates (np.ndarray): A 2D array of shape (M, 2) representing the candidate points.

    Returns:
        int: The index of the candidate point that is closest to the polygon vertices.
    """
    distances = []
    for candidate in candidates:
        distances.append(np.sum(np.sum((polygon_vertices - candidate) ** 2, axis=1)))

    return int(np.argmin(np.array(distances)))

def assign_cell_types(max_x: int, max_y: int, regions: list, vertices: np.ndarray, cells: list, target_density: float, center_only: bool = False) -> np.ndarray:
    """
    Assigns cell types based on the target density of multiciliated cells.

    Args:
        cells (list): List of cell data (x, y, neighbours, area).
        target_density (float): Target density of multiciliated cells.

    Returns:
        np.ndarray: Array indicating the type of each cell (1 for multiciliated, 0 for basic, 3 for border).
    """
    type_mask = np.zeros(len(cells))
    current_density = 0
    
    cell_points = np.array([np.array([cell[0], cell[1]]) for cell in cells])

    accessed_points = []
    for region in regions:
        polygon = vertices[region]
        for point in polygon:
            border_index = is_center(polygon, cell_points)
            accessed_points.append(border_index)
            if (abs(0 - point[0]) < 0.00001) or (abs(max_x - point[0]) < 0.00001) or  (abs(0 - point[1]) < 0.00001) or (abs(max_y - point[1]) < 0.00001):
                type_mask[border_index] = 3

    # FIXME: This has absolutely terrible efficiency and is bottom-of-the-barrel stuff
    for i in range(len(cell_points)):
        if i not in accessed_points:
            type_mask[i] = 3
   
    if center_only:
        area_center = np.array([max_x / 2, max_y / 2])
        center_distances = []
        for cell_point in cell_points:
            center_distances.append(np.sum((area_center - cell_point) ** 2))

        type_mask[int(np.argmin(np.array(center_distances)))] = 1
    else:
        while current_density < target_density:
            candidates = np.where(type_mask == 0)[0]
            if len(candidates) == 0:
                break

            chosen_cell = np.random.choice(candidates)
            type_mask[chosen_cell] = 1
            for cell_index in cells[chosen_cell][2]:
                if not type_mask[cell_index]:
                    type_mask[cell_index] = 2
            current_density = len(np.where(type_mask == 1)[0]) / len(cells)

        type_mask[np.where(type_mask == 2)[0]] = 0

    return type_mask

def calculate_intersection(a, b, c, d) -> np.ndarray:
    """
    Calculates the intersection point of two line segments ab and cd.

    Args:
        a (np.ndarray): The first point of the first line segment.
        b (np.ndarray): The second point of the first line segment.
        c (np.ndarray): The first point of the second line segment.
        d (np.ndarray): The second point of the second line segment.

    Returns:
        np.ndarray: The intersection point as a numpy array, or an empty array if there is no intersection.
    """
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    x4, y4 = d

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return np.array([])

    t_numerator = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    u_numerator = (x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)
    t = t_numerator / denominator
    u = u_numerator / denominator

    if 0 <= t <= 1 and 0 <= u <= 1:
        intersection = a + t * (b - a)
        return intersection
    else:
        return np.array([])

def isin(a, b, threshold):
    """
    Finds the indices of points in b that are within a threshold distance of point a.

    Args:
        a (np.ndarray): A point represented as a 1D array.
        b (np.ndarray): A 2D array of points to compare with.
        threshold (float): The distance threshold.

    Returns:
        np.ndarray: The indices of points in b that are within the threshold distance of point a.
    """

    return np.where(np.all(np.abs(b - a) < threshold, axis=1))[0]
    
def constrain_voronoi(max_x: int, max_y: int, voronoi: Voronoi):
    """
    Constrains the Voronoi diagram within a specified box.

    Args:
        max_x (int): The maximum x-coordinate of the bounding box.
        max_y (int): The maximum y-coordinate of the bounding box.
        voronoi (Voronoi): The Voronoi diagram to constrain.

    Returns:
        tuple: A tuple containing the constrained vertices and the regions.
    """
    region_points = []
    for region in voronoi.regions:
        if not region or -1 in region:
            continue
        
        new_region_points = []
        for i in range(len(region)):
            x, y = voronoi.vertices[region[i]]

            if i < len(region)-1:
                next_x, next_y = voronoi.vertices[region[i+1]]
            else:
                next_x, next_y = voronoi.vertices[region[0]]

            flag = True

            if (x < 0): # Left
                if (0 <= next_y <= max_y) and (0 <= next_x <= max_x):
                    intersection = calculate_intersection(np.array([x, y]), np.array([next_x, next_y]), [0, 0], [0, max_y])
                    if intersection.size != 0:
                        new_region_points.append(intersection)
                        flag = False
            elif (next_x < 0):
                if (0 <= y <= max_y) and (0 <= x <= max_x):
                    intersection = calculate_intersection(np.array([x, y]), np.array([next_x, next_y]), [0, 0], [0, max_y])
                    new_region_points.append(np.array([x, y]))
                    if intersection.size != 0:
                        new_region_points.append(intersection)
                        flag = False
            elif (x > max_x): # Right
                if (0 <= next_y <= max_y) and (0 <= next_x <= max_x):
                    intersection = calculate_intersection(np.array([x, y]), np.array([next_x, next_y]), [max_x, 0], [max_x, max_y])
                    if intersection.size != 0:
                        new_region_points.append(intersection)
                        flag = False
            elif (next_x > max_x):
                if (0 <= y <= max_y) and (0 <= x <= max_x):
                    intersection = calculate_intersection(np.array([x, y]), np.array([next_x, next_y]), [max_x, 0], [max_x, max_y])
                    new_region_points.append(np.array([x, y]))
                    if intersection.size != 0:
                        new_region_points.append(intersection)
                        flag = False

            if flag:
                if (y < 0): # Down 
                    if (0 <= next_y <= max_y) and (0 <= next_x <= max_x):
                        intersection = calculate_intersection(np.array([x, y]), np.array([next_x, next_y]), [0, 0], [max_x, 0])
                        if intersection.size != 0:
                            new_region_points.append(intersection)
                            flag = False
                elif (next_y < 0):
                    if (0 <= y <= max_y) and (0 <= x <= max_x):
                        intersection = calculate_intersection(np.array([x, y]), np.array([next_x, next_y]), [0, 0], [max_x, 0])
                        new_region_points.append(np.array([x, y]))
                        if intersection.size != 0:
                            new_region_points.append(intersection)
                            flag = False
                elif (y > max_y): # Up 
                    if (0 <= next_y <= max_y) and (0 <= next_x <= max_x):
                        intersection = calculate_intersection(np.array([x, y]), np.array([next_x, next_y]), [0, max_y], [max_x, max_y])
                        if intersection.size != 0:
                            new_region_points.append(intersection)
                            flag = False
                elif (next_y > max_y):
                    if (0 <= y <= max_y) and (0 <= x <= max_x):
                        intersection = calculate_intersection(np.array([x, y]), np.array([next_x, next_y]), [0, max_y], [max_x, max_y])
                        new_region_points.append(np.array([x, y]))
                        if intersection.size != 0:
                            new_region_points.append(intersection)
                            flag = False
            
            
            if 0 <= x <= max_x and 0 <= y <= max_y and flag:
                new_region_points.append(np.array([x, y]))

        region_points.append(new_region_points)

    # Consolidate points
    constrained_vertices = np.empty((0, 2), dtype=float)
    constrained_regions = []

    for region in region_points:
        region_map = []
        for point in region:
            if constrained_vertices.size > 0 :
                vertex_index = isin(point, constrained_vertices, 0.00001)
                if vertex_index.size == 0:
                    constrained_vertices = np.append(constrained_vertices, [point], axis=0)
                    region_map.append(len(constrained_vertices) - 1)
                else:
                    region_map.append(vertex_index[0])
            else:
                constrained_vertices = np.append(constrained_vertices, [point], axis=0)
                region_map.append(len(constrained_vertices) - 1)

        constrained_regions.append(region_map)

    return (constrained_vertices, constrained_regions)
