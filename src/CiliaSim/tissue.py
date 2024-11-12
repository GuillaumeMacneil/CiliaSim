from CiliaSim.jit_functions import (
    calculate_boundary_reflection,
    calculate_force_matrix,
    hexagonal_grid_layout,
)
from CiliaSim.plotting import (
    TissuePlot,
)

import numpy as np
from scipy.spatial import Voronoi, Delaunay, KDTree
from scipy.stats import qmc
import matplotlib.pyplot as plt
import json
from collections import defaultdict
from numba.typed import List
from tqdm import tqdm
import os
from scipy.sparse import csr_matrix


class Tissue:
    def __init__(
        self,
        x: int,
        y: int,
        cilia_density: float,
        centre_only: bool = False,
        plotting: bool = False,
        tracking: bool = False,
    ):
        if (x < 2) or (y < 2):
            raise ValueError("Tissue dimensions must be 2x2 or larger.")

        self.x = x
        self.y = y
        self.density = cilia_density
        self.num_cells = (x - 1) * (y - 1)
        self.global_iteration = 0
        self.center_only = centre_only
        self.plotting = plotting
        self.tracking = tracking
        self.plot = TissuePlot()
        self.plot_type = 0

        self.cell_points = np.array([], dtype=np.float64)
        self.cell_types = np.array([], dtype=int)
        self.target_areas = np.array([], dtype=np.float64)
        self.adjacency_matrix = np.zeros(
            (self.num_cells, self.num_cells), dtype=np.int64
        )
        self.force_matrix = np.zeros((self.num_cells, self.num_cells, 2))
        self.distance_matrix = np.zeros((self.num_cells, self.num_cells))
        self.boundary_cycle = np.array([], dtype=np.int64)
        self.voronoi = None

        self.target_spring_length = 1
        self.target_cell_area = np.sqrt(3) / 2
        self.critical_length_delta = 0.2

        self.net_energy = np.array([])

        # State encoding variables
        self.force_states = defaultdict(dict)
        self.cell_states = {}

        # Cilia force
        self.cilia_forces = {}

        # Exogenous flow force
        self.flow_force = np.array([0, 0])

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

        self.generate_cells()

    def hexagonal_grid_layout(self):
        self.cell_points = np.array(
            hexagonal_grid_layout(self.num_cells, self.x, self.y)
        )
        self.num_cells = len(self.cell_points)
        self.generate_cells()

    def generate_cells(self):
        delaunay = Delaunay(self.cell_points)
        indptr, indices = delaunay.vertex_neighbor_vertices

        # Create sparse matrix using CSR format
        rows = np.repeat(np.arange(self.num_cells), np.diff(indptr))
        cols = indices
        data = np.ones_like(indices)
        self.adjacency_matrix = csr_matrix(
            (data, (rows, cols)), shape=(self.num_cells, self.num_cells)
        )

        # 2. Initialize arrays
        self.cell_types = np.zeros(self.num_cells)

        # 3. Optimize boundary detection
        bounds = np.array(
            [
                [
                    self.cell_points[:, 0].min(),
                    self.cell_points[:, 1].max(),
                ],  # top-left
                [
                    self.cell_points[:, 0].max(),
                    self.cell_points[:, 1].max(),
                ],  # top-right
                [
                    self.cell_points[:, 0].max(),
                    self.cell_points[:, 1].min(),
                ],  # bottom-right
                [
                    self.cell_points[:, 0].min(),
                    self.cell_points[:, 1].min(),
                ],  # bottom-left
            ]
        )

        # Create edge points more efficiently
        num_edge_points = 50
        edges = []
        for i in range(4):
            start, end = bounds[i], bounds[(i + 1) % 4]
            edge_points = np.column_stack(
                [
                    np.linspace(start[0], end[0], num_edge_points),
                    np.linspace(start[1], end[1], num_edge_points),
                ]
            )
            edges.append(edge_points)

        # 4. Optimize boundary cell detection using KDTree
        kd_tree = KDTree(self.cell_points)
        edge_points = np.vstack(edges)
        distances, indices = kd_tree.query(edge_points)

        # Mark boundary cells
        self.cell_types[indices] = 1
        unique_boundary = np.unique(indices)
        self.boundary_cycle = np.append(self.boundary_cycle, unique_boundary)

        # 5. Optimize center/density-based cell assignment
        if self.center_only:
            area_center = np.array([self.x / 2, self.y / 2])
            center_distances = np.sum((self.cell_points - area_center) ** 2, axis=1)
            self.cell_types[np.argmin(center_distances)] = 2
        else:
            # Optimize density-based assignment
            non_boundary_mask = self.cell_types == 0
            target_count = int(self.density * self.num_cells)

            while np.sum(self.cell_types == 2) < target_count:
                candidates = np.where(non_boundary_mask)[0]
                if len(candidates) == 0:
                    break

                chosen_cell = np.random.choice(candidates)
                self.cell_types[chosen_cell] = 2

                # Get neighbors using sparse matrix
                neighbors = self.adjacency_matrix[chosen_cell].toarray().flatten()
                neighbor_mask = (neighbors > 0) & non_boundary_mask
                self.cell_types[neighbor_mask] = 3
                non_boundary_mask[neighbor_mask] = False

            # Reset temporary marked cells
            self.cell_types[self.cell_types == 3] = 0

        # 6. Optimize target areas assignment
        self.target_areas = np.zeros(self.num_cells)
        self.target_areas[self.cell_types != 1] = self.target_cell_area

        # 7. Final steps
        self.voronoi = Voronoi(self.cell_points)
        self.evaluate_boundary()
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
                self.force_states[self.global_iteration][
                    int(multiciliated_cells[i])
                ] = (unit_directions[i] * magnitude).tolist()
                if -1 not in self.force_states.keys():
                    self.force_states[self.global_iteration][
                        -1
                    ] = self.flow_force.tolist()

    def set_uniform_cilia_forces(self, direction: list, magnitude: float):
        force = np.array(direction) * magnitude
        multiciliated_cells = np.where(self.cell_types == 2)[0]
        for multiciliated_cell in multiciliated_cells:
            self.cilia_forces[multiciliated_cell] = force

            if self.tracking:
                self.force_states[self.global_iteration][
                    int(multiciliated_cell)
                ] = force.tolist()
                if -1 not in self.force_states.keys():
                    self.force_states[self.global_iteration][
                        -1
                    ] = self.flow_force.tolist()

    def set_flow(self, flow_direction, flow_magnitude):
        self.flow_force = np.array(flow_direction) * flow_magnitude

        if self.tracking:
            self.force_states[self.global_iteration][-1] = self.flow_force.tolist()
            if len(self.force_states[self.global_iteration].keys()) == 1:
                for key in self.cilia_forces.keys():
                    self.force_states[self.global_iteration][
                        int(key)
                    ] = self.cilia_forces[int(key)].tolist()

    def evaluate_boundary(self):
        # Determine connectivity from Voronoi map
        voronoi_neighbours = [set() for _ in range(self.num_cells)]
        for ridge in self.voronoi.ridge_points:
            i, j = ridge
            voronoi_neighbours[i].add(j)
            voronoi_neighbours[j].add(i)

        boundary_set = set(self.boundary_cycle)
        boundary_len = len(self.boundary_cycle)
        boundary_delete_list = []
        for i in range(-1, boundary_len - 1):
            current = self.boundary_cycle[i]
            before = self.boundary_cycle[i - 1]
            after = self.boundary_cycle[i + 1]

            # Ensure that boundary connectivity follows the cycle
            voronoi_neighbours[current] = voronoi_neighbours[current] - boundary_set
            voronoi_neighbours[current].update([before, after])

            # Remove excessive boundary cells
            if len(voronoi_neighbours[current]) <= 2:
                current_before = self.cell_points[before] - self.cell_points[current]
                current_after = self.cell_points[after] - self.cell_points[current]
                pair_dot = np.dot(current_before, current_after)
                angle = np.arccos(
                    pair_dot
                    / (np.linalg.norm(current_before) * np.linalg.norm(current_after))
                )

                if angle < np.pi / 2:
                    boundary_delete_list.append(current)

        # Add additional boundary cells
        edges = np.stack(
            (self.boundary_cycle, np.roll(self.boundary_cycle, shift=-1)), axis=1
        )
        new_boundary_cells = []
        for i in range(len(edges)):
            a, b = edges[i]
            shared_cells = voronoi_neighbours[a] & voronoi_neighbours[b]
            shared_non_boundary = shared_cells - boundary_set

            if len(shared_non_boundary) == 0:
                continue
            c = int(next(iter(shared_non_boundary)))

            a_point = self.cell_points[a]
            b_point = self.cell_points[b]
            c_point = self.cell_points[c]

            reflect_flag, reflected_point = calculate_boundary_reflection(
                a_point, b_point, c_point
            )

            if reflect_flag:
                self.cell_points = np.append(
                    self.cell_points, [reflected_point], axis=0
                )
                self.cell_types = np.append(self.cell_types, 1)
                self.target_areas = np.append(self.target_areas, 0)
                new_boundary_cells.append([i + 1, len(self.cell_points) - 1])
                self.num_cells += 1
                voronoi_neighbours.append({a, b, c})

        new_boundary_cells = reversed(new_boundary_cells)
        for new_boundary_cell in new_boundary_cells:
            self.boundary_cycle = np.insert(
                self.boundary_cycle, new_boundary_cell[0], new_boundary_cell[1]
            )

        # Construct an adjacency matrix
        self.adjacency_matrix = np.zeros(
            (self.num_cells, self.num_cells), dtype=np.int64
        )
        for i in range(self.num_cells):
            self.adjacency_matrix[i][list(voronoi_neighbours[i])] = 1

        # Handle the boundary cell deletions
        boundary_delete_list = sorted(boundary_delete_list, reverse=True)
        for deletion in boundary_delete_list:
            self.cell_points = np.delete(self.cell_points, deletion, axis=0)
            self.cell_types = np.delete(self.cell_types, deletion, axis=0)
            self.target_areas = np.delete(self.target_areas, deletion, axis=0)
            self.adjacency_matrix = np.delete(self.adjacency_matrix, deletion, axis=0)
            self.adjacency_matrix = np.delete(self.adjacency_matrix, deletion, axis=1)
            self.num_cells -= 1

            self.boundary_cycle = np.delete(
                self.boundary_cycle, self.boundary_cycle == deletion
            )
            greater = np.argwhere(self.boundary_cycle > deletion)
            self.boundary_cycle[greater] -= 1

            cilia_force_keys = list(self.cilia_forces.keys())
            new_cilia_forces = dict()
            for i in range(len(cilia_force_keys)):
                if cilia_force_keys[i] > deletion:
                    new_cilia_forces[cilia_force_keys[i] - 1] = self.cilia_forces[
                        cilia_force_keys[i]
                    ]
                else:
                    new_cilia_forces[cilia_force_keys[i]] = self.cilia_forces[
                        cilia_force_keys[i]
                    ]

            self.cilia_forces = new_cilia_forces

        self.voronoi = Voronoi(self.cell_points)

    def calculate_force_matrix(self):
        # FIXME: Declaration of these large Numba lists is SO slow
        # TODO i think if you have an array where the size is max of vertices.size, you can just use that
        voronoi_vertices = List(
            [
                self.voronoi.vertices[
                    self.voronoi.regions[self.voronoi.point_region[i]]
                ]
                for i in range(self.num_cells)
            ]
        )

        self.force_matrix, self.distance_matrix = calculate_force_matrix(
            self.num_cells,
            self.target_spring_length,
            self.critical_length_delta,
            self.cell_points,
            self.cell_types,
            self.target_areas,
            voronoi_vertices,
            self.adjacency_matrix,
        )

        # Calculate cilia and external force contributions
        for m_index in np.where(self.cell_types == 2)[0]:
            self.force_matrix[m_index, m_index] = (
                self.cilia_forces[m_index] + self.flow_force
            )

    def simulate(
        self,
        title: str,
        dt: float = 0.01,
        damping: float = 0.95,
        iterations: int = 5000,
        plot_frequency: int = 100,
        plotting: bool = True,
    ):
        """
        Simulate the tissue for a given number of iterations.

        Parameters
        ----------
        title : str
            The title of the simulation.
        dt : float, optional
            The time step of the simulation. The default is 0.01.
        damping : float, optional
            The damping factor of the simulation. The default is 0.95.
        iterations : int, optional
            The number of iterations to simulate. The default is 5000.
        plot_frequency : int, optional
            The frequency at which to plot the simulation. The default is 100.
        plotting : bool, optional
            Whether to plot the simulation. The default is True.

        Returns
        -------
        None
        """
        plt.ion()
        for _ in tqdm(range(iterations), desc=title):
            # TODO this is easily the most expensive part of the simulation
            self.evaluate_boundary()
            self.calculate_force_matrix()
            # TODO consider using velocity verlet for a more stable simulation (larger timesteps)
            total_force = np.sum(self.force_matrix, axis=0)
            self.cell_points += total_force * damping * dt
            self.cell_states[self.global_iteration] = self.cell_points

        plt.ioff()

    def write_to_file(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        json_data = {
            "parameters": {"x": self.x, "y": self.y, "cilia_density": self.density},
            "cell_types": self.cell_types.tolist(),
            "target_areas": self.target_areas.tolist(),
            "force_states": self.force_states,
            "cell_states": {k: v.tolist() for k, v in self.cell_states.items()},
            "net_energy": self.net_energy.tolist(),
        }

        json_object = json.dumps(json_data)

        with open(path, "w") as output_file:
            output_file.write(json_object)
