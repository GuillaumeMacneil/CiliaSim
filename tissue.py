from cell import BasicCell, MulticiliatedCell

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from collections import defaultdict
import matplotlib.pyplot as plt # <- TESTING

class Tissue():
    def __init__(self, size: int, cilia_density: float):
        self.size = size
        self.density = cilia_density
        self.num_cells = size ** 2
        self.cells = []

    def generate_cells(self, points: np.ndarray):
        # Generate a voronoi diagram to define the tissue
        voronoi = Voronoi(points)
        neighbours = defaultdict(list)
        for ridge_point in voronoi.ridge_points:
            if ridge_point[1] not in neighbours[ridge_point[0]]:
                neighbours[ridge_point[0]].append(ridge_point[1])

            if ridge_point[0] not in neighbours[ridge_point[1]]:
                neighbours[ridge_point[1]].append(ridge_point[0])

        # Store tissue state as a set of cells (basic / multiciliated)
        cell_data = []
        for i in range(len(points)):
            cell_data.append((points[i][0], points[i][1], np.array(neighbours[i])))
            
        cell_type_mask = assign_cell_types(cell_data, self.density)
        
        for i in range(len(cell_data)):
            if cell_type_mask[i]:
                self.cells.append(MulticiliatedCell(i, *cell_data[i]))
            else:
                self.cells.append(BasicCell(i, *cell_data[i]))
        
        self.cells = np.array(self.cells)
    
    def random_layout(self):
        x_coords = np.random.uniform(0, self.size, self.num_cells)
        y_coords = np.random.uniform(0, self.size, self.num_cells)
        self.generate_cells(np.column_stack((x_coords, y_coords)))

    def hexagonal_grid_layout(self):
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

    def render(self):
        type_mask = []
        points = []

        for cell in self.cells:
            if isinstance(cell, MulticiliatedCell):
                type_mask.append(1)
            else:
                type_mask.append(0)

            points.append((cell.x, cell.y))

        voronoi = Voronoi(points)
        _fig, ax = plt.subplots()
        voronoi_plot_2d(voronoi, ax=ax, show_vertices=False)
        
        for i in range(len(voronoi.point_region)):
            if type_mask[i]:
                region = voronoi.regions[voronoi.point_region[i]]
                # TODO: Fill the infinite vertices based on their bounding box intersections
                if not -1 in region:
                    polygon = [voronoi.vertices[j] for j in region]
                    ax.fill(*zip(*polygon), color='orange', alpha=0.4)      

        plt.show()


def assign_cell_types(cells: list, target_density: float) -> np.ndarray:
    type_mask = np.zeros(len(cells))
    current_density = 0

    while current_density < target_density:
        candidates = np.where(type_mask == 0)[0]
        if len(candidates) == 0:
            break

        chosen_cell = np.random.choice(candidates)
        type_mask[chosen_cell] = 1
        type_mask[cells[chosen_cell][2]] = 2
        current_density = len(np.where(type_mask == 1)[0]) / len(cells)

    type_mask[np.where(type_mask == 2)[0]] = 0

    return type_mask


# TESTING
tissue = Tissue(10, 0.01)
tissue.hexagonal_grid_layout()
tissue.render()
