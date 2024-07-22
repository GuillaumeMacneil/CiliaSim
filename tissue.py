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

        self.global_iteration = 0
        self.center_only = False
        self.plot = (None, None, None, None)
        self.plot_type = 0

        self.cell_points = np.array([])
        self.cell_types = np.array([])
        self.target_areas = np.array([])
        self.adjacency_matrix = np.zeros((self.num_cells, self.num_cells))
        self.force_matrix = np.zeros((self.num_cells, self.num_cells, 2))
        self.distance_matrix = np.zeros((self.num_cells, self.num_cells))

        self.target_spring_length = 1
        self.target_cell_area = 1
        self.critical_length_delta = 0.5

        # State encoding variables
        self.tracking = False
        self.force_states = {} # [[Global Iteration, Affected Cell (-1 indicates all), Direction [x, y], Magnitude z], ...]
        self.cell_states = {} # [[Cell Positions], [Cell Positions], ...]
        
        # Exogenous flow 
        self.flow_direction = np.array([0, 0])
        self.flow_magnitude = 0
        
    def set_center_only(self, value: bool):
        self.center_only = value

    def set_tracking(self):
        self.tracking = True

    def random_layout(self):
        points = []
        radius = 1 / np.sqrt(self.num_cells)
        while len(points) != self.num_cells:
            radius = radius * 0.95
            pd_sampler = qmc.PoissonDisk(d=2, radius=radius)
            points = pd_sampler.random(n=self.num_cells) * [self.x - 1, self.y - 1]
            points += 0.5

        self.cell_points = np.array(points)
        self.num_cells = len(self.cell_points)

        # This is a hacky method of (almost always) ensuring that the random tissue is constrained by boundary points
        self.generate_cells(specialize=False)        
        self.simulate("Random Layout Annealing (Tension Only, No Specialization)", 1500, 100, tension_only=True)
        self.generate_cells()        
        self.simulate("Random Layout Annealing (Tension Only)", 1500, 100, tension_only=True)
        self.global_iteration = 0

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

        self.cell_points = np.array(points)
        self.num_cells = len(self.cell_points)
        self.generate_cells()


    def generate_cells(self, specialize: bool = True):
        delaunay = Delaunay(self.cell_points)
        neighbour_vertices = delaunay.vertex_neighbor_vertices

        self.adjacency_matrix = np.zeros((self.num_cells, self.num_cells))
        self.cell_types = np.zeros(self.num_cells)

        for i in range(self.num_cells):
            neighbours = neighbour_vertices[1][neighbour_vertices[0][i]:neighbour_vertices[0][i+1]]
            self.adjacency_matrix[i][neighbours] = 1

        if specialize:
            voronoi = Voronoi(self.cell_points)

            for i in range(self.num_cells):
                region_index = voronoi.point_region[i]
                if -1 in voronoi.regions[region_index]:
                   self.cell_types[i] = 1

            if self.center_only:
                area_center = np.array([self.x / 2, self.y / 2])
                center_distances = []
                for cell_point in self.cell_points:
                    center_distances.append(np.sum((area_center - cell_point) ** 2))

                self.cell_types[int(np.argmin(np.array(center_distances)))] = 2
            else:
                current_density = 0
                while current_density < self.density:
                    candidates = np.where(self.cell_types == 0)[0]
                    if len(candidates) == 0:
                        break

                    chosen_cell = np.random.choice(candidates)
                    self.cell_types[chosen_cell] = 2
                    for cell_index in np.where(self.adjacency_matrix[chosen_cell] == 1):
                        if not self.cell_types[cell_index]:
                            self.cell_types[cell_index] = 3

                    current_density = len(np.where(self.cell_types == 2)[0]) / self.num_cells

                self.cell_types[np.where(self.cell_types == 3)[0]] = 0

            self.evaluate_boundary()

            #target_area = 0
            #count = 0
            #for i in range(len(cell_points)):
            #    if type_mask[i] != 1:
            #        region_index = voronoi.point_region[i]
            #        target_area += polygon_area(voronoi.vertices[voronoi.regions[region_index]])
            #        count += 1
            #target_area /= count

            self.target_areas = np.zeros(self.num_cells) 
            for i in range(self.num_cells):
                if self.cell_types[i] != 1:
                    self.target_areas[i] = self.target_cell_area

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

    def set_plot_area_deltas(self):
        self.plot_type = 5

    def calculate_force_matrix(self, tension_only: bool = False):
        delaunay = Delaunay(self.cell_points)
        voronoi = Voronoi(self.cell_points)

        neighbour_vertices = delaunay.vertex_neighbor_vertices

        boundary_cells = np.where(self.cell_types == 1)
        self.adjacency_matrix = np.zeros((self.num_cells, self.num_cells))
        for i in range(self.num_cells):
            neighbours = neighbour_vertices[1][neighbour_vertices[0][i]:neighbour_vertices[0][i+1]]
            self.adjacency_matrix[i, neighbours] = 1

        delete_list = []
        for cell in boundary_cells[0]:
            cell_neighbours = np.where(self.adjacency_matrix[cell] == 1)
            boundary_neighbours = np.intersect1d(cell_neighbours, boundary_cells)
            
            if len(cell_neighbours) == len(boundary_neighbours):
                delete_list.append(cell)
                for neighbour in boundary_neighbours:
                    self.adjacency_matrix[neighbour, cell] = 0

        for cell in delete_list:
            self.cell_points = np.delete(self.cell_points, cell)
            self.cell_types = np.delete(self.cell_types, cell)
            self.adjacency_matrix = np.delete(self.adjacency_matrix, cell, axis=0)
            self.adjacency_matrix = np.delete(self.adjacency_matrix, cell, axis=1)
            self.num_cells -= 1

        boundary_cells = np.setdiff1d(boundary_cells, delete_list)

        for cell in boundary_cells:
            cell_neighbours = np.where(self.adjacency_matrix[cell] == 1)
            boundary_neighbours = np.intersect1d(cell_neighbours, boundary_cells)
            differences = self.cell_points[boundary_neighbours] - self.cell_points[cell]
            distances = np.linalg.norm(differences, axis=1)
            sorted_boundary_neighbours = boundary_neighbours[np.argsort(distances)]
            furthest_neighbours = sorted_boundary_neighbours[2:]

            for neighbour in furthest_neighbours:
                self.adjacency_matrix[cell, neighbour] = 0
                self.adjacency_matrix[neighbour, cell] = 0
        
        self.force_matrix = np.zeros((self.num_cells, self.num_cells, 2))
        self.distance_matrix = np.zeros((self.num_cells, self.num_cells))
      
        for i in range(self.num_cells):
            # Calculate spring forces
            neighbours = np.where(self.adjacency_matrix[i] == 1)
            neighbour_positions = self.cell_points[neighbours]
            differences = neighbour_positions - self.cell_points[i]
            distances = np.linalg.norm(differences, axis=1)
            unit_vectors = differences / distances[:, np.newaxis]

            force_magnitudes = distances[:, np.newaxis] - self.target_spring_length

            self.force_matrix[i, neighbours] += force_magnitudes * unit_vectors
            self.distance_matrix[i, neighbours] = distances

            if self.cell_types[i] == 1 and not tension_only:
                self.force_matrix[i, neighbours] /= 10
            
            if self.cell_types[i] != 1 and not tension_only:
                # Calculate pressure forces
                region_index = voronoi.point_region[i]
                area = polygon_area(voronoi.vertices[voronoi.regions[region_index]])
                area_difference = self.target_areas[i] - area

                self.force_matrix[i, neighbours] += area_difference * unit_vectors
                self.force_matrix[neighbours, i] += area_difference * unit_vectors

            if self.cell_types[i] == 2 and not tension_only:
                # Calculate external force contributions
                flow = self.flow_direction * self.flow_magnitude
                
                pinv_differences = np.linalg.pinv(differences.T)
                force_distribution = np.dot(pinv_differences, flow)

                external_forces = force_distribution[:, np.newaxis] * differences
                self.force_matrix[i, neighbours] += external_forces 
       
    def evaluate_connectivity(self):
        self.adjacency_matrix = np.zeros((self.num_cells, self.num_cells))

        delaunay = Delaunay(self.cell_points)
        neighbour_vertices = delaunay.vertex_neighbor_vertices
        for i in range(self.num_cells):
            neighbours = neighbour_vertices[1][neighbour_vertices[0][i]:neighbour_vertices[0][i+1]]
            self.adjacency_matrix[i, neighbours] = 1

        critical_edges = np.where(self.distance_matrix - self.target_spring_length > self.critical_length_delta)
        if critical_edges[0].size == 0:
            return None

        edge_pairs = np.column_stack(critical_edges)
        sorted_pairs = np.sort(edge_pairs, axis=1)
        unique_pairs = np.unique(sorted_pairs, axis=0)

        for a, b in unique_pairs:
            shared_cells = np.intersect1d(np.where(self.adjacency_matrix[a] == 1)[0], np.where(self.adjacency_matrix[b] == 1)[0])
            if len(shared_cells) == 2:
                #new_length = np.linalg.norm(shared_cells[1] - shared_cells[0])
                #if new_length < self.distance_matrix[a, b]:
                self.adjacency_matrix[a, b] = 0
                self.adjacency_matrix[b, a] = 0
                self.adjacency_matrix[shared_cells[0], shared_cells[1]] = 1
                self.adjacency_matrix[shared_cells[1], shared_cells[0]] = 1

    def evaluate_boundary(self):
        # Determine which cells are boundary cells and the edges between them
        boundary_cells = np.where(self.cell_types == 1)
        visited = []
        edges = []
        for i in boundary_cells[0]:
            for neighbour in np.where(self.adjacency_matrix[i] == 1)[0]:
                if neighbour in boundary_cells[0] and neighbour not in visited:
                    edges.append((i, neighbour))
                
                visited.append(i)

        # Add additional boundary cells
        for a, b in edges:
            shared_cells = np.intersect1d(np.where(self.adjacency_matrix[a] == 1)[0], np.where(self.adjacency_matrix[b] == 1)[0])
            shared_non_boundary = np.setdiff1d(shared_cells, boundary_cells)
            if len(shared_non_boundary) == 0:
                continue
            c = int(shared_non_boundary[0])

            a_point = self.cell_points[a]
            b_point = self.cell_points[b]
            c_point = self.cell_points[c]

            ac = c_point - a_point
            bc = c_point - b_point
            ac_dot_bc = np.dot(ac, bc)
            angle = np.arccos(ac_dot_bc / (np.linalg.norm(ac) * np.linalg.norm(bc)))

            if angle > np.pi / 2:
                edge_vector = b_point - a_point
                edge_unit_vector = edge_vector / np.linalg.norm(edge_vector)

                projection_length = np.dot(c_point - a_point, edge_unit_vector)
                projection_vector = projection_length * edge_unit_vector

                reflected_point = 2 * (a_point + projection_vector) - c_point
                
                self.cell_points = np.append(self.cell_points, [reflected_point], axis=0)
                self.cell_types = np.append(self.cell_types, 1)
                self.target_ares = np.append(self.target_areas, 0)
                
                new_row = np.zeros(self.num_cells)
                new_row[[a, b, c]] = 1
                self.adjacency_matrix = np.vstack([self.adjacency_matrix, new_row])
                self.num_cells += 1
                new_column = np.zeros(self.num_cells)
                new_column[[a, b, c]] = 1
                self.adjacency_matrix = np.hstack([self.adjacency_matrix, new_column[:, np.newaxis]])

    def increment_global_iteration(self, title: str, x_lim: int = 0, y_lim: int = 0, plot_frequency: int = 100):
        if self.global_iteration % plot_frequency == 0:
            information = f"Iteration: {self.global_iteration}\nCilia force magnitude: {self.flow_magnitude}\nCilia force direction: {self.flow_direction}"
            if self.plot_type == 0:
                self.plot = plot_tissue(self.cell_points, self.cell_types, title, 0.5, self.plot, x_lim=x_lim, y_lim=y_lim, information=information)
            elif self.plot_type == 1:
                self.plot = plot_springs(self.cell_points, self.cell_types, self.adjacency_matrix, title, 0.5, self.plot, x_lim=x_lim, y_lim=y_lim, information=information)
            elif self.plot_type == 2:
                self.plot = plot_force_vectors(self.cell_points, self.cell_types, self.force_matrix, title, 0.5, self.plot, x_lim=x_lim, y_lim=y_lim, information=information)
            elif self.plot_type == 3:
                self.plot = plot_major_axes(self.cell_points, self.cell_types, title, 0.5, self.plot, x_lim=x_lim, y_lim=y_lim, information=information)
            elif self.plot_type == 4:
                self.plot = plot_avg_major_axes(self.cell_points, self.cell_types, self.adjacency_matrix, title, 0.5, self.plot, x_lim=x_lim, y_lim=y_lim, information=information)
            elif self.plot_type == 5:
                self.plot = plot_area_delta(self.cell_points, self.cell_types, self.target_cell_area, title, 0.5, self.plot, information=information)

        self.global_iteration += 1

    def simulate(self, title: str, iterations: int = 5000, plot_frequency: int = 100, tension_only: bool = False):
        plt.ion()
        for i in range(iterations):
            self.calculate_force_matrix(tension_only)
            total_force = np.sum(self.force_matrix, axis=1)
            self.cell_points += total_force * 0.95 * 0.01

            if self.tracking:
                self.cell_states[self.global_iteration] = self.cell_points.tolist()

            self.increment_global_iteration(title, x_lim=self.x, y_lim=self.y, plot_frequency=plot_frequency)
            #self.evaluate_connectivity()
            self.evaluate_boundary()
            
    def write_to_file(self, path: str):
        json_data = {"parameters": {"x": self.x, "y": self.y, "cilia_density": self.density}, "cell_types": self.cell_types.tolist(), "target_areas": self.target_areas.tolist(), "force_states": self.force_states, "cell_states": self.cell_states}
        json_object = json.dumps(json_data)

        with open(path, "w") as output_file:
            output_file.write(json_object)
