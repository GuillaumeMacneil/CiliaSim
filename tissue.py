from cell import BasicCell, MulticiliatedCell, BorderCell
from functions import *

import numpy as np
from scipy.spatial import Voronoi
from scipy.stats import qmc
from collections import defaultdict
import matplotlib.pyplot as plt
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
            x (int): The size x-dimension of the tissue.
            y (int): The size y-dimension of the tissue.
            cilia_density (float): The density of multiciliated cells.
        """

        if (x < 2) or (y < 2):
            raise ValueError("Tissue dimensions must be 2x2 or larger.")

        self.x = x
        self.y = y
        self.density = cilia_density
        self.num_cells = (x - 1) * (y - 1)
        self.cells = []
        self.center_only = False

    def set_center_only(self, value: bool):
        self.center_only = value

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
            
        cell_type_mask = assign_cell_types(self.x, self.y, constrained_regions, constrained_vertices, cell_data, self.density, self.center_only)
        
        for i in range(len(cell_data)):
            if cell_type_mask[i] == 1:
                self.cells.append(MulticiliatedCell(i, *cell_data[i]))
            elif cell_type_mask[i] == 3:
                self.cells.append(BorderCell(i, *cell_data[i]))
            else:
                self.cells.append(BasicCell(i, *cell_data[i]))
        
        #self.cells = np.array(self.cells)
    
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

    def render(self, title: str = "", plot: tuple = (None, None)):
        """
        Renders the tissue with cells colored based on their type.

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

        if not plot[0] and not plot[1]:
            fig, ax = plt.subplots()
        else:
            fig, ax = plot
            ax.clear()

        constrained_vertices, constrained_regions = constrain_voronoi(self.x, self.y, voronoi)

        for i in range(len(constrained_regions)):
            polygon = constrained_vertices[constrained_regions[i]]
            ax.fill(*zip(*polygon), edgecolor="black", fill=False)
            

        basic_points = np.array(basic_points)
        multiciliated_points = np.array(multiciliated_points)
        border_points = np.array(border_points)

        #ax.scatter(basic_points[:, 0], basic_points[:, 1], s=20, color="blue")
        ax.scatter(multiciliated_points[:, 0], multiciliated_points[:, 1], s=20, color="orange")   
        ax.scatter(border_points[:, 0], border_points[:, 1], s=20, color="green")   

        ax.plot([0, self.x, self.x, 0, 0], [0, 0, self.y, self.y, 0], 'k-')

        plt.title(title)
        plt.show()
        plt.pause(0.1)

        return (fig, ax)

    def anneal(self, iterations: int = 2000):
        plt.ion()
        plot = (None, None)
        for iteration in range(iterations):
            for cell in self.cells:
                cell.step(self.cells)

            if iteration % 50 == 0:
                plot = self.render(f"Tissue @ iteration {iteration}", plot)


# TESTING
tissue = Tissue(10, 10, 0.1)
tissue.set_center_only(True)
tissue.random_layout()
tissue.anneal()
