from cell import BasicCell, MulticiliatedCell, BorderCell

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import qmc
from collections import defaultdict
import matplotlib.pyplot as plt # <- TESTING

class Tissue():
    """
    Represents a tissue consisting of multiciliated and basic cells arranged in a 2D space.

    Attributes:
        size (int): The size of the tissue (length of one side of the square tissue).
        density (float): The density of multiciliated cells in the tissue.
        num_cells (int): The total number of cells in the tissue.
        cells (np.ndarray): An array of cells in the tissue.
    """

    def __init__(self, x: int, y: int, cilia_density: float):
        """
        Initializes a Tissue object with a given size and cilia density.

        Args:
            size (int): The size of the tissue.
            cilia_density (float): The density of multiciliated cells.
        """

        if (x < 2) or (y < 2):
            raise ValueError("Tissue dimensions must be 2x2 or larger.")

        self.x = x
        self.y = y
        self.density = cilia_density
        self.num_cells = (x - 1) * (y - 1)
        self.cells = []

    def generate_cells(self, points: np.ndarray):
        """
        Generates cells based on given points and assigns them as multiciliated or basic.

        Args:
            points (np.ndarray): Array of points representing cell positions.
        """
        voronoi = Voronoi(points)
        neighbours = defaultdict(list)
        areas = defaultdict(float)

        # Constrain to tissue size
        constrained_vertices, constrained_regions = constrain_voronoi(self.x, self.y, voronoi)

        # Calculate neighbours
        for ridge_point in voronoi.ridge_points:
            if ridge_point[1] not in neighbours[ridge_point[0]]:
                neighbours[ridge_point[0]].append(ridge_point[1])

            if ridge_point[0] not in neighbours[ridge_point[1]]:
                neighbours[ridge_point[1]].append(ridge_point[0])

        # Calculate areas
        for point_index, region_index in enumerate(voronoi.point_region):
            region = voronoi.regions[region_index]
            if -1 not in region:
                polygon = constrained_vertices[region]
                area = polygon_area(polygon)
                areas[point_index] = area
            else:
                areas[point_index] = np.inf

        cell_data = []
        for i in range(len(points)):
            cell_data.append((points[i][0], points[i][1], np.array(neighbours[i]), areas[i]))

        cell_type_mask = assign_cell_types(self.x, self.y, constrained_regions, constrained_vertices, cell_data, self.density)
        
        for i in range(len(cell_data)):
            if cell_type_mask[i] == 1:
                self.cells.append(MulticiliatedCell(i, *cell_data[i]))
            elif cell_type_mask[i] == 3:
                self.cells.append(BorderCell(i, *cell_data[i]))
            else:
                self.cells.append(BasicCell(i, *cell_data[i]))
        
        self.cells = np.array(self.cells)
    
    def random_layout(self):
        """
        Generates a Poisson-disk sampled layout of cells within the tissue to ensure even distribution.
        """
        points = []
        radius = 1 / np.sqrt(self.num_cells)
        while len(points) != self.num_cells:
            radius = radius * 0.95
            pd_sampler = qmc.PoissonDisk(d=2, radius=radius)
            points = pd_sampler.random(n=self.num_cells) * [self.x - 1, self.y - 1]
            points += 0.5

        points = np.array(points)
        self.generate_cells(points)

    def hexagonal_grid_layout(self):
        """
        Generates a hexagonal grid layout of cells within the tissue.
        """
        hexagon_side_length = np.sqrt(1 / (3 * np.sqrt(3) / 2))

        horizontal_spacing = 3 * hexagon_side_length
        vertical_spacing = np.sqrt(3) * hexagon_side_length

        points = []
        row = 0
        while row * vertical_spacing < self.y - 1:
            for col in range(int(self.x - 1 / horizontal_spacing) + 1):
                x = col * horizontal_spacing + (row % 2) * (1.5 * hexagon_side_length)
                y = row * vertical_spacing
                if x < self.x - 1 and y < self.y - 1:
                    points.append((x + 0.5, y + 0.5))
            row += 1

        if len(points) > self.num_cells:
            points = points[:self.num_cells]

        self.generate_cells(np.array(points))

    def render(self, title: str = ""):
        """
        Renders the tissue with cells colored based on their type.
        Multiciliated cells are orange, border cells are green, and basic cells are blue.

        Args:
            title (str): The title of the plot.
        """
        multiciliated_points = []
        border_points = []
        basic_points = []
        points = []

        for cell in self.cells:
            if isinstance(cell, MulticiliatedCell):
                multiciliated_points.append(np.array([cell.x, cell.y]))
            elif isinstance(cell, BorderCell):
                border_points.append(np.array([cell.x, cell.y]))
            else:
                basic_points.append(np.array([cell.x, cell.y]))

            points.append(np.array([cell.x, cell.y]))

        voronoi = Voronoi(points)

        fig, ax = plt.subplots()

        constrained_vertices, constrained_regions = constrain_voronoi(self.x, self.y, voronoi)

        for region in constrained_regions:
            points = constrained_vertices[region]
            plt.fill(*zip(*points), edgecolor="black", fill=False)

        basic_points = np.array(basic_points)
        multiciliated_points = np.array(multiciliated_points)
        border_points = np.array(border_points)

        ax.scatter(basic_points[:, 0], basic_points[:, 1], s=20, color="blue")
        ax.scatter(multiciliated_points[:, 0], multiciliated_points[:, 1], s=20, color="orange")   
        ax.scatter(border_points[:, 0], border_points[:, 1], s=20, color="green")   

        #ax.plot([0, self.x, self.x, 0, 0], [0, 0, self.y, self.y, 0], 'k-')

        plt.title(title)
        plt.show()


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
    distances = []
    for candidate in candidates:
        distances.append(np.sum(np.sum((polygon_vertices - candidate) ** 2, axis=1)))

    return int(np.argmin(np.array(distances)))

def assign_cell_types(max_x: int, max_y: int, regions: list, vertices: np.ndarray, cells: list, target_density: float) -> np.ndarray:
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
    return np.where(np.all(np.abs(b - a) < threshold, axis=1))[0]
    
def constrain_voronoi(max_x: int, max_y: int, voronoi: Voronoi):
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


# TESTING
tissue = Tissue(10, 10, 0.1)
tissue.random_layout()
tissue.render("Test Plot")
