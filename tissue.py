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

    def __init__(self, size: int, cilia_density: float):
        """
        Initializes a Tissue object with a given size and cilia density.

        Args:
            size (int): The size of the tissue.
            cilia_density (float): The density of multiciliated cells.
        """
        self.size = size
        self.density = cilia_density
        self.num_cells = size ** 2
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

        for ridge_point in voronoi.ridge_points:
            if ridge_point[1] not in neighbours[ridge_point[0]]:
                neighbours[ridge_point[0]].append(ridge_point[1])

            if ridge_point[0] not in neighbours[ridge_point[1]]:
                neighbours[ridge_point[1]].append(ridge_point[0])

        for point_index, region_index in enumerate(voronoi.point_region):
            region = voronoi.regions[region_index]
            if -1 not in region:
                polygon = voronoi.vertices[region]
                area = polygon_area(polygon)
                areas[point_index] = area
            else:
                areas[point_index] = np.inf

        cell_data = []
        for i in range(len(points)):
            cell_data.append((points[i][0], points[i][1], np.array(neighbours[i]), areas[i]))
            
        cell_type_mask = assign_cell_types(cell_data, self.density)
        
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
        radius = self.size / self.num_cells
        while len(points) != self.num_cells:
            radius = radius * 0.9
            pd_sampler = qmc.PoissonDisk(d=2, radius=radius)
            points = pd_sampler.random(n=self.num_cells) * [self.size, self.size]

        points = np.array(points)
        self.generate_cells(points)

    def hexagonal_grid_layout(self):
        """
        Generates a hexagonal grid layout of cells within the tissue.
        """
        single_cell_area = (self.size ** 2) / self.num_cells
        radius = np.sqrt(2 * single_cell_area / (3 * np.sqrt(3)))
        cell_width = radius * np.sqrt(3)
        cell_height = radius * 2
        
        num_cells_x = int(np.ceil(self.size / cell_width))
        num_cells_y = int(np.ceil(self.size / (cell_height * 0.75)))

        points = []
        for i in range(num_cells_y):
            for j in range(num_cells_x):
                x = j * cell_width
                y = i * (cell_height * 0.75)
                
                if i % 2 == 1:
                    x += cell_width * 0.5
                
                points.append((x, y))

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
        points = []

        for cell in self.cells:
            if isinstance(cell, MulticiliatedCell):
                multiciliated_points.append((cell.x, cell.y))
            elif isinstance(cell, BorderCell):
                border_points.append((cell.x, cell.y))

            points.append((cell.x, cell.y))

        voronoi = Voronoi(points)
        _, ax = plt.subplots()
        voronoi_plot_2d(voronoi, ax=ax, show_vertices=False, show_points=False)

        other_points = np.array([a for a in points if (a not in multiciliated_points) or (a not in border_points)])
        multiciliated_points = np.array(multiciliated_points)
        border_points = np.array(border_points)

        ax.scatter(other_points[:, 0], other_points[:, 1], s=20, color="blue")
        ax.scatter(multiciliated_points[:, 0], multiciliated_points[:, 1], s=20, color="orange")   
        ax.scatter(border_points[:, 0], border_points[:, 1], s=20, color="green")   

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

def assign_cell_types(cells: list, target_density: float) -> np.ndarray:
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
    
    avg_area = 0
    for cell in cells:
        if cell[3] != np.inf:
            avg_area += cell[3]
    avg_area /= len(cells)

    # TODO: I need to think of a smarter way of determining the border cells
    for i in range(len(cells)):
        if cells[i][3] > 0.7 * avg_area:
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


# TESTING
tissue = Tissue(10, 0.1)
tissue.random_layout()
tissue.render("Test Plot")
