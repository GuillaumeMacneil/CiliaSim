from functions import *
from plotting import *

import numpy as np
from scipy.spatial import Voronoi, Delaunay, KDTree
from scipy.stats import qmc
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import json
from collections import defaultdict

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
        self.plotting = False
        self.plot = TissuePlot()
        self.plot_type = 0

        self.cell_points = np.array([])
        self.cell_types = np.array([])
        self.target_areas = np.array([])
        self.adjacency_matrix = np.zeros((self.num_cells, self.num_cells), dtype=np.int32)
        self.force_matrix = np.zeros((self.num_cells, self.num_cells, 2))
        self.distance_matrix = np.zeros((self.num_cells, self.num_cells))
        self.boundary_cycle = np.array([], dtype=np.int32)
        self.voronoi = None

        self.target_spring_length = 1
        self.target_cell_area = np.sqrt(3) / 2
        self.critical_length_delta = 0.2

        self.net_energy = np.array([])

        # State encoding variables
        self.tracking = False
        self.force_states = defaultdict(dict)
        self.cell_states = {}
        
        # Cilia force
        self.cilia_forces = {}

        # Exogenous flow force
        self.flow_force = np.array([0, 0])
        
    def set_center_only(self, value: bool):
        self.center_only = value

    def set_tracking(self):
        self.tracking = True

    def set_plotting(self):
        self.plotting = True

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
        self.simulate("Random Layout Annealing (Tension Only, No Specialization)", 1500, 100, tension_only=True, plotting=self.plotting)
        self.generate_cells()        
        self.set_plot_boundary_cycle()
        self.simulate("Random Layout Annealing", 1500, 100, plotting=self.plotting)
        self.set_plot_basic()
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
            x_min, y_min = np.min(self.cell_points, axis=0) 
            x_max, y_max = np.max(self.cell_points, axis=0) 

            kd_tree = KDTree(self.cell_points)
            num_edge_comparison_points = 50

            top_edge = np.linspace([x_min, y_max], [x_max, y_max], num_edge_comparison_points)
            right_edge = np.linspace([x_max, y_max], [x_max, y_min], num_edge_comparison_points)
            bottom_edge = np.linspace([x_max, y_min], [x_min, y_min], num_edge_comparison_points)
            left_edge = np.linspace([x_min, y_min], [x_min, y_max], num_edge_comparison_points)

            for edge in [top_edge, right_edge, bottom_edge, left_edge]:
                for comparison_point in edge:
                    _, point_index = kd_tree.query(comparison_point)
                    self.cell_types[point_index] = 1
                    if not np.any(self.boundary_cycle == point_index):
                        self.boundary_cycle = np.append(self.boundary_cycle, point_index)

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
                    for cell_index in np.where(self.adjacency_matrix[chosen_cell] == 1)[0]:
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

            self.set_uniform_cilia_forces([0, 0], 0)
   
    def add_cilia_force(self, cell_index: int, force: list):
        self.cilia_forces[cell_index] = np.array(force)

    def set_random_cilia_forces(self, magnitude: float):
        multiciliated_cells = np.where(self.cell_types == 2)[0]
        non_unit_directions = np.random.uniform(-1, 1, [len(multiciliated_cells), 2])
        unit_directions = non_unit_directions / np.linalg.norm(non_unit_directions)
        for i in range(len(multiciliated_cells)):
            self.cilia_forces[multiciliated_cells[i]] = unit_directions[i] * magnitude

            if self.tracking:
                self.force_states[self.global_iteration][int(multiciliated_cells[i])] = (unit_directions[i] * magnitude).tolist()
                if -1 not in self.force_states.keys():
                    self.force_states[self.global_iteration][-1] = self.flow_force.tolist()

    def set_uniform_cilia_forces(self, direction: list, magnitude: float):
        force = np.array(direction) * magnitude
        multiciliated_cells = np.where(self.cell_types == 2)[0]
        for multiciliated_cell in multiciliated_cells:
            self.cilia_forces[multiciliated_cell] = force

            if self.tracking:
                self.force_states[self.global_iteration][int(multiciliated_cell)] = force.tolist()
                if -1 not in self.force_states.keys():
                    self.force_states[self.global_iteration][-1] = self.flow_force.tolist()

    def set_flow(self, flow_direction, flow_magnitude):
        self.flow_force = np.array(flow_direction) * flow_magnitude

        if self.tracking:
            self.force_states[self.global_iteration][-1] = self.flow_force.tolist()
            if len(self.force_states[self.global_iteration].keys()) == 1:
                for key in self.cilia_forces.keys():
                    self.force_states[self.global_iteration][int(key)] = self.cilia_forces[int(key)].tolist()

    def set_plot_basic(self):
        self.plot_type = 0

    def set_plot_spring(self):
        self.plot_type = 1

    def set_plot_force_vector_rel(self):
        self.plot_type = 2

    def set_plot_force_vector_abs(self):
        self.plot_type = 3

    def set_plot_major_axes(self):
        self.plot_type = 4

    def set_plot_avg_major_axes(self):
        self.plot_type = 5

    def set_plot_area_deltas(self):
        self.plot_type = 6

    def set_plot_neighbour_histogram(self):
        self.plot_type = 7

    def set_plot_shape_factor_histogram(self):
        self.plot_type = 8

    def set_plot_anisotropy_histogram(self):
        self.plot_type = 9

    def set_plot_Q_divergence(self):
        self.plot_type = 10

    def set_plot_boundary_cycle(self):
        self.plot_type = 11

    def calculate_force_matrix(self, tension_only: bool = False):
        self.voronoi = Voronoi(self.cell_points)

        boundary_cells = np.where(self.cell_types == 1)
        self.adjacency_matrix = np.zeros((self.num_cells, self.num_cells))
        for ridge in self.voronoi.ridge_points:
            i, j = ridge
            self.adjacency_matrix[i, j] = 1
            self.adjacency_matrix[j, i] = 1

        csr_adjacency = csr_matrix(self.adjacency_matrix)

        #delete_list = []
        #for cell in boundary_cells[0]:
        #    cell_neighbours = np.where(self.adjacency_matrix[cell] == 1)
        #    boundary_neighbours = np.intersect1d(cell_neighbours, boundary_cells)
            
        #    if len(cell_neighbours) == len(boundary_neighbours):
        #        delete_list.append(cell)
        #        for neighbour in boundary_neighbours:
        #            self.adjacency_matrix[neighbour, cell] = 0

        #for cell in delete_list:
        #    self.cell_points = np.delete(self.cell_points, cell)
        #    self.cell_types = np.delete(self.cell_types, cell)
        #    self.adjacency_matrix = np.delete(self.adjacency_matrix, cell, axis=0)
        #    self.adjacency_matrix = np.delete(self.adjacency_matrix, cell, axis=1)
        #    self.num_cells -= 1
#
        #boundary_cells = np.setdiff1d(boundary_cells, delete_list)

        boundary_cells = boundary_cells[0]
        """ for cell in boundary_cells:
            cell_neighbours = np.where(self.adjacency_matrix[cell] == 1)
            boundary_neighbours = np.intersect1d(cell_neighbours, boundary_cells)
            differences = self.cell_points[boundary_neighbours] - self.cell_points[cell]
            distances = np.linalg.norm(differences, axis=1)
            sorted_boundary_neighbours = boundary_neighbours[np.argsort(distances)]
            furthest_neighbours = sorted_boundary_neighbours[2:]

            for neighbour in furthest_neighbours:
                self.adjacency_matrix[cell, neighbour] = 0
                self.adjacency_matrix[neighbour, cell] = 0 """
        
        magnitude_matrix = np.zeros((self.num_cells, self.num_cells))
        self.distance_matrix = np.zeros((self.num_cells, self.num_cells))
        unit_vector_matrix = np.zeros((self.num_cells, self.num_cells, 2))

        net_energy = 0 
        visited = []
        for i in range(self.num_cells):
            # Calculate spring forces
            #neighbours = np.nonzero(self.adjacency_matrix[i])
            neighbours = csr_adjacency.indices[csr_adjacency.indptr[i]:csr_adjacency.indptr[i+1]]
            neighbour_positions = self.cell_points[neighbours]
            differences = neighbour_positions - self.cell_points[i]
            distances = np.linalg.norm(differences, axis=1)
            unit_vectors = differences / distances[:, np.newaxis]

            self.distance_matrix[i, neighbours] = distances
            unit_vector_matrix[i, neighbours] = unit_vectors

            """ energy_distances = distances[~np.isin(neighbours, visited)[0]]
            energy_contribution = 0.5 * (energy_distances - self.target_spring_length) ** 2
            net_energy += np.sum(energy_contribution) """
            visited.append(i)

            force_magnitudes = self.target_spring_length - distances

            magnitude_matrix[i, neighbours] += force_magnitudes

        magnitude_matrix = np.clip(magnitude_matrix, -self.critical_length_delta, None)

        if not tension_only:
            for i in np.where(self.cell_types == 1)[0]:
                #neighbours = np.nonzero(self.adjacency_matrix[i])
                neighbours = csr_adjacency.indices[csr_adjacency.indptr[i]:csr_adjacency.indptr[i+1]]
                magnitude_matrix[i, neighbours] /= 10
            
            for i in np.where(self.cell_types != 1)[0]:
                # Calculate pressure forces
                #neighbours = np.nonzero(self.adjacency_matrix[i])
                neighbours = csr_adjacency.indices[csr_adjacency.indptr[i]:csr_adjacency.indptr[i+1]]
                area = polygon_area(self.voronoi.vertices[self.voronoi.regions[self.voronoi.point_region[i]]])
                area_difference = (self.target_areas[i] - area)
                net_energy += 0.5 * area_difference ** 2

                magnitude_matrix[i, neighbours] += area_difference / len(neighbours)
                magnitude_matrix[neighbours, i] += area_difference / len(neighbours)

        self.force_matrix = magnitude_matrix.T[:, :, np.newaxis] * unit_vector_matrix

        # Calculate cilia and external force contributions
        for m_index in np.where(self.cell_types == 2)[0]:
            self.force_matrix[m_index, m_index] = self.cilia_forces[m_index] + self.flow_force          

        if not tension_only:
            self.net_energy = np.append(self.net_energy, net_energy)

    def calculate_shape_factors(self):
        non_boundary_cells = np.where(self.cell_types != 1)[0]
        shape_factors = []
        for cell in non_boundary_cells:
            vertices = self.voronoi.vertices[self.voronoi.regions[self.voronoi.point_region[cell]]]
            area = polygon_area(vertices)
            perimeter = polygon_perimeter(vertices)
            shape_factors.append(perimeter / np.sqrt(area))
            
        return np.array(shape_factors)

    def evaluate_boundary(self):
        if self.boundary_cycle.size == 0:
            return 

        # FIXME: Determine which boundary points are unnecessary
        #kd_tree = KDTree(self.cell_points[self.boundary_cycle])
        #differ_mask = np.zeros(len(self.boundary_cycle))
        #for i in range(len(self.boundary_cycle)):
        #    _, indices = kd_tree.query(self.cell_points[self.boundary_cycle[i]], k=3)
        #    closest = self.boundary_cycle[indices[1:]]

        #    if i == 0:
        #        lr_points = [self.boundary_cycle[len(self.boundary_cycle) - 1], self.boundary_cycle[i+1]]
        #    elif i == len(self.boundary_cycle) - 1:
        #        lr_points = [self.boundary_cycle[i-1], self.boundary_cycle[0]]
        #    else:   
        #        lr_points = [self.boundary_cycle[i-1], self.boundary_cycle[i+1]]
            
        #    if closest[0] not in lr_points or closest[1] not in lr_points:
        #        print(f"{lr_points} -> {closest}")
            
        # Add additional boundary cells
        edges = np.stack((self.boundary_cycle, np.roll(self.boundary_cycle, shift=-1)),axis=1)
        new_boundary_cells = []
        csr_adjacency = csr_matrix(self.adjacency_matrix)

        all_neighbours = [set(csr_adjacency.indices[csr_adjacency.indptr[i]:csr_adjacency.indptr[i+1]]) for i in range(csr_adjacency.shape[0])]

        for i in range(len(edges)):
            a, b = edges[i]
            shared_cells = all_neighbours[a] & all_neighbours[b]
            shared_non_boundary = shared_cells - set(self.boundary_cycle)
            if len(shared_non_boundary) == 0:
                continue
            c = int(next(iter(shared_non_boundary)))

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
                new_boundary_cells.append([i+1, len(self.cell_points) - 1])
                
                new_row = np.zeros(self.num_cells)
                new_row[[a, b, c]] = 1
                self.adjacency_matrix = np.vstack([self.adjacency_matrix, new_row])
                self.num_cells += 1
                new_column = np.zeros(self.num_cells)
                new_column[[a, b, c]] = 1
                self.adjacency_matrix = np.hstack([self.adjacency_matrix, new_column[:, np.newaxis]])

        new_boundary_cells = reversed(new_boundary_cells)
        for new_boundary_cell in new_boundary_cells:
            self.boundary_cycle = np.insert(self.boundary_cycle, new_boundary_cell[0], new_boundary_cell[1])

    def increment_global_iteration(self, title: str, x_lim: int = 0, y_lim: int = 0, plot_frequency: int = 100):
        if self.global_iteration % plot_frequency == 0:
            information = f"Iteration: {self.global_iteration}\nEx. Flow Force: {self.flow_force}"
            if self.plot_type == 0:
                plot_tissue(self.cell_points, self.cell_types, title, 0.5, self.plot, x_lim=x_lim, y_lim=y_lim, information=information)
            elif self.plot_type == 1:
                plot_springs(self.cell_points, self.cell_types, self.adjacency_matrix, title, 0.5, self.plot, x_lim=x_lim, y_lim=y_lim, information=information)
            elif self.plot_type == 2:
                plot_force_vectors_rel(self.cell_points, self.cell_types, self.force_matrix, title, 0.5, self.plot, x_lim=x_lim, y_lim=y_lim, information=information)
            elif self.plot_type == 3:
                plot_force_vectors_abs(self.cell_points, self.cell_types, self.force_matrix, title, 0.5, self.plot, x_lim=x_lim, y_lim=y_lim, information=information)
            elif self.plot_type == 4:
                plot_major_axes(self.cell_points, self.cell_types, title, 0.5, self.plot, x_lim=x_lim, y_lim=y_lim, information=information)
            elif self.plot_type == 5:
                plot_avg_major_axes(self.cell_points, self.cell_types, self.adjacency_matrix, title, 0.5, self.plot, x_lim=x_lim, y_lim=y_lim, information=information)
            elif self.plot_type == 6:
                plot_area_delta(self.cell_points, self.cell_types, self.target_cell_area, title, 0.5, self.plot, information=information)
            elif self.plot_type == 7:
                plot_neighbour_histogram(self.adjacency_matrix, title, 0.5, self.plot, information=information)
            elif self.plot_type == 8:
                plot_shape_factor_histogram(self.calculate_shape_factors(), title, 0.5, self.plot, information=information)
            elif self.plot_type == 9:
                plot_anisotropy_histogram(self.cell_points, self.cell_types, title, 0.5, self.plot, information=information)
            elif self.plot_type == 10:
                plot_Q_divergence(self.cell_points, self.cell_types, title, 0.5, self.plot, information=information)
            elif self.plot_type == 11:
                plot_boundary_cycle(self.cell_points, self.cell_types, self.boundary_cycle, title, 0.5, self.plot, information=information)

        self.global_iteration += 1

    def simulate(self, title: str, iterations: int = 5000, plot_frequency: int = 100, tension_only: bool = False, plotting: bool = True):
        plt.ion()
        for i in range(iterations):
            self.calculate_force_matrix(tension_only)
            total_force = np.sum(self.force_matrix, axis=0)
            self.cell_points += total_force * 0.95 * 0.01

            if self.tracking:
                self.cell_states[self.global_iteration] = self.cell_points.tolist()

            if plotting:
                self.increment_global_iteration(title, x_lim=self.x, y_lim=self.y, plot_frequency=plot_frequency)
            else:
                self.global_iteration += 1

            self.evaluate_boundary()
            
    def write_to_file(self, path: str):
        json_data = {"parameters": {"x": self.x, "y": self.y, "cilia_density": self.density}, "cell_types": self.cell_types.tolist(), "target_areas": self.target_areas.tolist(), "force_states": self.force_states, "cell_states": self.cell_states, "net_energy": self.net_energy.tolist()}
        json_object = json.dumps(json_data)

        with open(path, "w") as output_file:
            output_file.write(json_object)
