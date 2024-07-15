from cell import BasicCell, MulticiliatedCell, BorderCell
from functions import *
from plotting import *

import numpy as np
from scipy.spatial import Voronoi, Delaunay
from scipy.stats import qmc
from collections import defaultdict
import matplotlib.pyplot as plt
import json

class Tissue():
    def __init__(self, x: int, y: int, cilia_density: float):
        if (x < 2) or (y < 2):
            raise ValueError("Tissue dimensions must be 2x2 or larger.")

        self.x = x
        self.y = y
        self.density = cilia_density
        self.num_cells = (x - 1) * (y - 1)
        self.cells = []
        self.global_iteration = 0
        self.center_only = False
        self.plot = (None, None, None)
        self.plot_type = 0

        self.force_matrix = np.zeros((self.num_cells, self.num_cells, 2))
        self.position_buffer = []

        self.target_spring_length = 1

        # State encoding variables
        self.tracking = False
        self.cell_inits = [] # [[Cell Type (0 -> Basic, 1 -> Boundary, 2 -> Multiciliated), Target Area], ...]
        self.force_states = {} # [[Global Iteration, Affected Cell (-1 indicates all), Direction [x, y], Magnitude z], ...]
        self.cell_states = {} # [[Cell Positions], [Cell Positions], ...]
        
        # Exogenous flow and torque parameters
        self.flow_direction = np.array([0, 0])
        self.flow_magnitude = 0
        
    def set_center_only(self, value: bool):
        self.center_only = value

    def set_tracking(self):
        self.tracking = True

    def generate_cells(self, points: np.ndarray):
        voronoi = Voronoi(points)
        neighbours = defaultdict(list)

        for ridge_point in voronoi.ridge_points:
            if ridge_point[1] not in neighbours[ridge_point[0]]:
                neighbours[ridge_point[0]].append(ridge_point[1])

            if ridge_point[0] not in neighbours[ridge_point[1]]:
                neighbours[ridge_point[1]].append(ridge_point[0])

        for i in range(len(points)):
            self.cells.append(BasicCell(i, points[i][0], points[i][1], np.array(neighbours[i])))
    
    def random_layout(self):
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
        num_rings = int(np.floor(1/2 + np.sqrt(12 * self.num_cells - 3) / 6))

        points = []
        cx = self.x / 2
        cy = self.y / 2

        # Add the center point
        points.append((cx, cy))
        
        for i in range(1, num_rings + 1):
            for j in range(6 * i):
                angle = j * np.pi / (3 * i)
                if i % 2 == 0:
                    angle += np.pi / (3 * i)
                x = cx + i * np.cos(angle)
                y = cy + i * np.sin(angle)
                points.append((x, y))
                    
        self.generate_cells(np.array(points))

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

        for i in range(len(type_mask)):
            id = self.cells[i].id
            x = self.cells[i].x
            y = self.cells[i].y
            neighbours = self.cells[i].neighbours
            if type_mask[i] == 1:
                self.cells[i] = BorderCell(id, x, y, neighbours)
            elif type_mask[i] == 2:
                self.cells[i] = MulticiliatedCell(id, x, y, neighbours)

        self.evaluate_boundary()

        target_area = 0
        count = 0
        for i in range(len(cell_points)):
            if type_mask[i] != 1:
                target_area += polygon_area(cell_points[self.cells[i].neighbours])
                count += 1

        target_area /= count

        for i in range(len(type_mask)):
            if type_mask[i] == 0 or type_mask[i] == 2:
                self.cells[i].set_area(target_area)

        if self.tracking:
            for cell in self.cells:
                if isinstance(cell, BasicCell):
                    self.cell_inits.append([0, cell.area])
                elif isinstance(cell, BorderCell):
                    self.cell_inits.append([1, 0])
                else:
                    self.cell_inits.append([2, cell.area])

    def set_flow(self, flow_direction, flow_magnitude):
        self.flow_direction = np.array(flow_direction)
        self.flow_magnitude = flow_magnitude

        if self.tracking:
            self.force_states[self.global_iteration] = [-1, flow_direction, flow_magnitude]

    def set_plot_basic(self):
        self.plot_type = 0

    def set_plot_spring(self):
        self.plot_type = 1

    def set_plot_force_vector(self):
        self.plot_type = 2

    def set_plot_major_axes(self):
        self.plot_type = 3

    def set_plot_avg_major_axes(self):
        self.plot_type = 4

    def calculate_force_matrix(self, tension_only: bool = False):
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
                forces = ((distances[:, np.newaxis] - self.target_spring_length) * unit_vectors) / 10
            else:
                forces = (distances[:, np.newaxis] - self.target_spring_length) * unit_vectors

            if not tension_only:
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
                self.cell_inits.append([1, 0])
                self.position_buffer.append([reflected_point[0], reflected_point[1]])

    def increment_global_iteration(self, title: str):
        if self.global_iteration % 100 == 0:
            information = f"Iteration: {self.global_iteration}\nCilia force magnitude: {self.flow_magnitude}\nCilia force direction: {self.flow_direction}"
            if self.plot_type == 0:
                self.plot = plot_tissue(self.cells, title, 0.5, self.plot, information=information)
            elif self.plot_type == 1:
                self.plot = plot_springs(self.cells, title, 0.5, self.plot, information=information)
            elif self.plot_type == 2:
                self.plot = plot_force_vectors(self.cells, self.force_matrix, title, 0.5, self.plot, information=information)
            elif self.plot_type == 3:
                self.plot = plot_major_axes(self.cells, title, 0.5, self.plot, information=information)
            elif self.plot_type == 4:
                self.plot = plot_avg_major_axes(self.cells, title, 0.5, self.plot, information=information)

        self.global_iteration += 1

    def simulate(self, title: str, iterations: int = 5000, annealing: bool = False):
        plt.ion()
        if annealing:
            tension_only_iterations = int(np.floor(iterations / 2))
            for i in range(tension_only_iterations):
                self.force_matrix = self.calculate_force_matrix(tension_only=True)
                for cell in self.cells:
                    cell.step(self.force_matrix)

                self.evaluate_boundary()
                self.increment_global_iteration(title)

            self.set_cell_types()

            for i in range(iterations - tension_only_iterations):
                self.force_matrix = self.calculate_force_matrix()
                for cell in self.cells:
                    cell.step(self.force_matrix)

                self.evaluate_boundary()
                self.increment_global_iteration(title)
        else:
            for i in range(iterations):
                self.force_matrix = self.calculate_force_matrix()
                for cell in self.cells:
                    cell.step(self.force_matrix)
                    self.position_buffer.append([cell.x, cell.y])

                self.evaluate_boundary()
                if self.tracking:
                    self.cell_states[self.global_iteration] = self.position_buffer
                self.position_buffer = []
                self.increment_global_iteration(title)
    
    def write_to_file(self, path: str):
        json_data = {"parameters": {"x": self.x, "y": self.y, "cilia_density": self.density}, "cell_inits": self.cell_inits, "force_states": self.force_states, "cell_states": self.cell_states}
        json_object = json.dumps(json_data)

        with open(path, "w") as output_file:
            output_file.write(json_object)
