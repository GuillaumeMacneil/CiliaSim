from cell import BasicCell, MulticiliatedCell, BorderCell
from functions import *

import numpy as np
from scipy.spatial import Voronoi, Delaunay
from scipy.stats import qmc
from collections import defaultdict
import matplotlib.pyplot as plt

import cProfile

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
        self.plot = (None, None)

        # Exogenous flow and torque parameters
        self.flow_direction = np.array([0, 0])
        self.flow_magnitude = 0
        self.torque_magnitude = 0

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

        # Calculate neighbours
        for ridge_point in voronoi.ridge_points:
            if ridge_point[1] not in neighbours[ridge_point[0]]:
                neighbours[ridge_point[0]].append(ridge_point[1])

            if ridge_point[0] not in neighbours[ridge_point[1]]:
                neighbours[ridge_point[1]].append(ridge_point[0])

        for i in range(len(points)):
            self.cells.append(BasicCell(i, points[i][0], points[i][1], np.array(neighbours[i])))
            
    
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

        # Plot cell edges
        if len(border_points) == 0:
            for i in range(len(voronoi.regions)):
                if -1 not in voronoi.regions[i]:
                    polygon = voronoi.vertices[voronoi.regions[i]]
                    ax.fill(*zip(*polygon), edgecolor="black", fill=False)
        else:
            for i in range(len(self.cells)):
                if not isinstance(self.cells[i], BorderCell):
                    region_index = voronoi.point_region[i]
                    polygon = voronoi.vertices[voronoi.regions[region_index]]
                    ax.fill(*zip(*polygon), edgecolor="black", fill=False)

            
        # Plot springs
        #for i in range(len(points)):
        #    start = points[i]
        #    for neighbour_index in self.cells[i].neighbours:
        #        end = points[neighbour_index]
        #        ax.plot([start[0], end[0]], [start[1], end[1]], color="blue")

        # Plot center points
        basic_points = np.array(basic_points)
        multiciliated_points = np.array(multiciliated_points)
        border_points = np.array(border_points)

        ax.scatter(basic_points[:, 0], basic_points[:, 1], s=20, color="blue")
        if multiciliated_points.size > 0 and border_points.size > 0:
            ax.scatter(multiciliated_points[:, 0], multiciliated_points[:, 1], s=20, color="orange")   
            ax.scatter(border_points[:, 0], border_points[:, 1], s=20, color="green")   

        # Plot border
        ax.plot([0, self.x, self.x, 0, 0], [0, 0, self.y, self.y, 0], 'k-')
        plt.title(title)
        plt.show()

        plt.xlim([0, self.x])
        plt.ylim([0, self.y])

        plt.pause(0.2)
        
        ax.clear()

        return (fig, ax)

    def set_cell_types(self):
        type_mask = np.zeros(len(self.cells))
        cell_points = np.array([np.array([cell.x, cell.y]) for cell in self.cells])
        voronoi = Voronoi(cell_points)

        for i in range(len(cell_points)):
            region_index = voronoi.point_region[i]
            if -1 in voronoi.regions[region_index]:
                type_mask[i] = 1

        if self.center_only:
            area_center = np.array([self.x / 2, self.y / 2])
            center_distances = []
            for cell_point in cell_points:
                center_distances.append(np.sum((area_center - cell_point) ** 2))

            type_mask[int(np.argmin(np.array(center_distances)))] = 2
        else:
            current_density = 0
            while current_density < self.density:
                candidates = np.where(type_mask == 0)[0]
                if len(candidates) == 0:
                    break

                chosen_cell = np.random.choice(candidates)
                type_mask[chosen_cell] = 2
                for cell_index in self.cells[chosen_cell].neighbours:
                    if not type_mask[cell_index]:
                        type_mask[cell_index] = 3

                current_density = len(np.where(type_mask == 2)[0]) / len(self.cells)

            type_mask[np.where(type_mask == 3)[0]] = 0

        target_area = 0
        count = 0
        for i in range(len(cell_points)):
            region_id = voronoi.point_region[i]
            if -1 not in voronoi.regions[region_id]:
                target_area += polygon_area(cell_points[self.cells[i].neighbours])
                count += 1

        target_area = target_area / count

        for i in range(len(type_mask)):
            id = self.cells[i].id
            x = self.cells[i].x
            y = self.cells[i].y
            neighbours = self.cells[i].neighbours
            if type_mask[i] == 0:
                #self.cells[i].set_area(polygon_area(voronoi.vertices[voronoi.regions[voronoi.point_region[id]]]))
                self.cells[i].set_area(target_area)
            elif type_mask[i] == 1:
                self.cells[i] = BorderCell(id, x, y, neighbours)
            elif type_mask[i] == 2:
                self.cells[i] = MulticiliatedCell(id, x, y, neighbours)
                #self.cells[i].set_area(polygon_area(voronoi.vertices[voronoi.regions[voronoi.point_region[id]]]))
                self.cells[i].set_area(target_area)

    def set_flow(self, flow_direction, flow_magnitude):
        self.flow_direction = flow_direction
        self.flow_magnitude = flow_magnitude

    def set_torque(self, torque_magnitude):
        self.torque_magnitude = torque_magnitude

    def calculate_force_matrix(self, annealing = False):
        cell_points = np.array([np.array([cell.x, cell.y]) for cell in self.cells])
        delaunay = Delaunay(cell_points)

        neighbour_vertices = delaunay.vertex_neighbor_vertices

        boundary_cells = []
        for i in range(len(self.cells)):
            neighbours = neighbour_vertices[1][neighbour_vertices[0][i]:neighbour_vertices[0][i+1]]
            self.cells[i].neighbours = neighbours
            if isinstance(self.cells[i], BorderCell):
                boundary_cells.append(i)

        for cell in boundary_cells:
            cell_neighbours = self.cells[cell].neighbours
            boundary_neighbours = np.intersect1d(cell_neighbours, boundary_cells)
            differences = cell_points[boundary_neighbours] - cell_points[cell]
            distances = np.linalg.norm(differences, axis=1)
            sorted_boundary_neighbours = boundary_neighbours[np.argsort(distances)]
            furthest_neighbours = sorted_boundary_neighbours[2:]

            for neighbour in furthest_neighbours:
                self.cells[cell].neighbours = np.setdiff1d(self.cells[cell].neighbours, neighbour)
                self.cells[neighbour].neighbours = np.setdiff1d(self.cells[neighbour].neighbours, cell)
        
        force_matrix = np.zeros((len(self.cells), len(self.cells), 2))

        for i in range(len(cell_points)):
            # Calculate attractive forces
            neighbour_positions = cell_points[self.cells[i].neighbours]
            differences = neighbour_positions - cell_points[i]
            distances = np.linalg.norm(differences, axis=1)
            unit_vectors = differences / distances[:, np.newaxis]

            if isinstance(self.cells[i], BorderCell):
                forces = ((distances[:, np.newaxis] - 1) * unit_vectors) / 10
            else:
                forces = (distances[:, np.newaxis] - 1) * unit_vectors

            if not annealing:
                # Calculate repulsive forces
                if not isinstance(self.cells[i], BorderCell):
                    area = polygon_area(cell_points[self.cells[i].neighbours])
                    area_difference = self.cells[i].area - area

                    forces -= (area_difference / len(unit_vectors)) * unit_vectors

                if isinstance(self.cells[i], MulticiliatedCell):
                    # Calculate external force contributions
                    flow = self.flow_direction * self.flow_magnitude
                    
                    pinv_differences = np.linalg.pinv(differences.T)
                    force_distribution = np.dot(pinv_differences, flow)

                    external_forces = force_distribution[:, np.newaxis] * differences
                    forces += external_forces 

            force_matrix[i, self.cells[i].neighbours] += forces 

        return force_matrix 

    def evaluate_boundary(self):
        # Determine which cells are boundary cells and the edges between them
        boundary_cells = []
        edges = []
        for i in range(len(self.cells)):
            if isinstance(self.cells[i], BorderCell):
                for neighbour in self.cells[i].neighbours:
                    if isinstance(self.cells[neighbour], BorderCell) and neighbour not in boundary_cells:
                        edges.append((i, neighbour))
                
                boundary_cells.append(i)

        # Add additional boundary cells
        for edge in edges:
            shared_cell = set(self.cells[edge[0]].neighbours).intersection(set(self.cells[edge[1]].neighbours))
            shared_cell = shared_cell.difference(boundary_cells)
            if len(shared_cell) == 0:
                continue
            shared_cell = shared_cell.pop()

            a = np.array([self.cells[shared_cell].x, self.cells[shared_cell].y])
            b = np.array([self.cells[edge[0]].x, self.cells[edge[0]].y])
            c = np.array([self.cells[edge[1]].x, self.cells[edge[1]].y])
            ab = a - b
            ac = a - c
            ab_dot_ac = np.dot(ab, ac)
            angle = np.arccos(ab_dot_ac / (np.linalg.norm(ab) * np.linalg.norm(ac)))
            
            if angle > np.pi / 2:
                edge_vector = c - b
                edge_unit_vector = edge_vector / np.linalg.norm(edge_vector)

                projection_length = np.dot(a - b, edge_unit_vector)
                projection_vector = projection_length * edge_unit_vector

                reflected_point = 2 * (b + projection_vector) - a

                self.cells.append(BorderCell(len(self.cells), reflected_point[0], reflected_point[1], np.array([edge[0], edge[1], shared_cell])))
                print(f"Adding new BorderCell at ({reflected_point[0]}, {reflected_point[1]})")

    def anneal(self, iterations: int = 2000):
        plt.ion()
        for iteration in range(iterations):
            force_matrix = self.calculate_force_matrix(annealing=True)
            for i in range(len(self.cells)):
                self.cells[i].step(force_matrix, annealing=True)

            if iteration % 100 == 0:
                self.plot = self.render(f"Tissue annealing @ iteration {iteration}", self.plot)

        self.set_cell_types()
        self.evaluate_boundary()

    def simulate(self, iterations: int = 5000):
        for iteration in range(iterations):
            force_matrix = self.calculate_force_matrix()
            for cell in self.cells:
                cell.step(force_matrix)

            self.evaluate_boundary()

            if iteration % 100 == 0:
                self.plot = self.render(f"Tissue under exogenous flow ({self.flow_magnitude}) @ iteration {iteration}", self.plot)


# TESTING
def main():
    tissue = Tissue(10, 10,  0.1)
    tissue.set_center_only(True)
    tissue.random_layout()
    tissue.anneal()
    tissue.set_flow(np.array([0, 1]), 0.4)
    tissue.simulate(3000)


cProfile.run('main()', sort="tottime")
#main()
