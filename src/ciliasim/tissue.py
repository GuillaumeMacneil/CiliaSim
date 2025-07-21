import numpy as np
import json
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from ciliasim import geometry 
from ciliasim import boundary 
from ciliasim import connectivity

class Tissue():
    def __init__(
            self, 
            x: int, 
            y: int, 
            density: float,
            spring_length: float = 1.0,
            critical_delta: float = 0.2,
            oversize_factor: float = 1.2,
            max_degree: int = 15,
            center_only: bool = False,
            random_layout: bool = False,
            save: bool = False,
            save_freq: int = 10,
            output_dir: str = ""
            ):
        # Dimension and layout parameters
        self.x = x
        self.y = y
        self.num_cells = (x - 1) * (y - 1)
        self.max_cells = int(self.num_cells * oversize_factor)
        self.max_degree = max_degree
       
        if (x < 2) or (y < 2):
            raise ValueError("Tissue dimensions must be 2x2 or larger.")

        # Mechanical parameters
        self.spring_length = spring_length
        self.critical_delta = critical_delta

        # Composition parameters
        self.density = density
        self.center_only = center_only

        # State saving parameters
        self.iteration = 0
        self.save = save
        self.save_freq = save_freq
        if not all([save, save_freq, output_dir]):
            raise ValueError("`save`, `save_freq` and `output_dir` must be provided for state saving.")
        else:
            self.output_path = Path(output_dir) / generate_filename(x, y, center_only, random_layout)
            self.output_path.mkdir(parents=True, exist_ok=False)
       
        # Positional state arrays
        self.cell_points = np.full((self.max_cells, 2), -1, np.float32)
        if random_layout:
            self.cell_points = geometry.uniform_random_layout(self.x, self.y, self.num_cells, self.max_cells)
            self.num_cells = self.cell_points.shape[0]
        else:
            self.cell_points = geometry.hexagonal_layout(self.x, self.y, self.num_cells, self.max_cells)
            self.num_cells = self.cell_points.shape[0]

        self.cell_types = np.full(self.max_cells, -1, dtype=np.int8)
        self.target_areas = np.zeros(self.max_cells, dtype=np.float32)
        self.adjacency = np.full((self.max_cells, self.max_degree), -1, dtype=np.int32)
        self.max_triangles = (2 * self.max_cells) - 5
        self.triangles = np.full((self.max_triangles, 3), -1, dtype=np.int32)
        self.boundary_cycle_mask = np.zeros(self.max_cells, dtype=bool)

        # Force state arrays
        self.internal_forces = np.zeros((self.max_cells, self.max_degree, 2), dtype=np.float32)
        self.cilia_forces = np.zeros((self.max_cells, 2), dtype=np.float32)
        self.flow_force = np.zeros(2, dtype=np.float32)

    def specify_cells(self, area_distribution: tuple[float, float]):
        # Define adjacency list using a Delaunay triangulation
        self.adjacency, self.triangles = connectivity.full_update(self.num_cells, self.max_cells, self.max_degree, self.max_triangles, self.cell_points)

        # Determine the boundary cycle and specify the individual cell types
        self.boundary_cycle_mask = boundary.create_boundary(self.cell_points, self.num_cells, 50)
        self.cell_types[:self.num_cells] = self.boundary_cycle_mask

        if self.center_only:
            area_center = np.array([self.x / 2, self.y / 2])
            center_distances = np.sum((area_center - self.cell_points) ** 2)
            self.cell_types[np.argmin(center_distances)] = 2
        else:
            current_density = 0
            while current_density < self.density:
                candidates = np.where(self.cell_types[:self.num_cells] == 0)[0]
                if candidates.size  == 0:
                    break

                chosen_cell = np.random.choice(candidates)
                self.cell_types[chosen_cell] = 2
                for cell_index in np.where(self.adjacency[chosen_cell] > 0)[0]:
                    if self.cell_types[cell_index] == 0:
                        self.cell_types[cell_index] = 3

                current_density = len(np.where(self.cell_types == 2)[0]) / self.num_cells

            self.cell_types[np.where(self.cell_types == 3)[0]] = 0
           
        # Assign each cell a target area
        if area_distribution[1] == 0:
            self.target_areas = geometry.uniform_target_areas(self.cell_types, area_distribution[0])
        else:
            self.target_areas = geometry.normal_random_target_areas(self.cell_types, self.num_cells, area_distribution[0], area_distribution[1])

        self.evaluate_boundary()
        
    def set_uniform_ciliary_forces(self, direction: np.ndarray, magnitude: float):
        force = direction * magnitude
        multiciliated_cells = np.where(self.cell_types == 2)[0]
        self.cilia_forces[multiciliated_cells] = force

    def set_random_ciliary_forces(self, magnitude: float):
        multiciliated_cells = np.where(self.cell_types == 2)[0]
        non_unit_directions = np.random.uniform(-1, 1, [len(multiciliated_cells), 2])
        unit_directions = non_unit_directions / np.linalg.norm(non_unit_directions)
        self.cilia_forces[multiciliated_cells] = unit_directions * magnitude

    def set_flow_force(self, direction: np.ndarray, magnitude: float):
        self.flow_force = direction * magnitude


    def evaluate_boundary(self):
        # Remove any unnecessary cells on the boundary and add additional cells if needed
        changed = False
        prev_num_cells = self.num_cells
        boundary.remove_excessive_boundary_cells(self.adjacency, self.boundary_cycle_mask, self.cell_points, self.cell_types, self.target_areas, self.num_cells)
        changed = prev_num_cells != self.num_cells
        boundary.add_additional_boundary_cells(self.adjacency, self.boundary_cycle_mask, self.cell_points, self.cell_types, self.target_areas, self.num_cells)
        changed = (prev_num_cells != self.num_cells) or changed

        # If cells have either been deleted, added or both, perform a new triangulation, otherwise perform a partial re-triangulation
        if changed:
            self.adjacency, self.triangles = connectivity.full_update(self.num_cells, self.max_cells, self.max_degree, self.max_triangles, self.cell_points)
        else:
            # FIXME: This should properly implement a partial update
            self.adjacency, self.triangles = connectivity.full_update(self.num_cells, self.max_cells, self.max_degree, self.max_triangles, self.cell_points)

    def simulate(self, iterations: int):
        for i in tqdm(range(iterations)):
            # Calculate internal forces
            self.internal_forces = calculate_forces(
                    self.num_cells,
                    self.max_cells,
                    self.max_degree,
                    self.spring_length,
                    self.critical_delta,
                    self.cell_points,
                    self.cell_types,
                    self.target_areas,
                    self.triangles,
                    self.adjacency
                    )
            # NOTE: May be backwards - I really hope not
            internal_force = np.sum(self.internal_forces, axis=1)

            # Introduce external forces and move cell centers accordingly
            multiciliated_mask = self.cell_types == 2
            total_force = internal_force + self.cilia_forces
            total_force[multiciliated_mask] += self.flow_force
            # FIXME: Should figure out what these magic numbers are
            self.cell_points += total_force * 0.95 * 0.01
            self.evaluate_boundary()

            if self.save:
                # Add static parameters header
                if self.iteration == 0:
                    with open(self.output_path / "params.json", "w") as params_file:
                        static_params = {
                                "x": self.x,
                                "y": self.y,
                                "density": self.density,
                                "spring_length": self.spring_length,
                                "critical_delta": self.critical_delta,
                                "max_degree": self.max_degree,
                                "max_cells": self.max_cells,
                                "max_triangles": self.max_triangles
                                }
                        json.dump(static_params, params_file)

                # Save state to file with given frequency
                if self.iteration % self.save_freq == 0:
                    circumcenters = geometry.calculate_circumcenters(self.cell_points, self.triangles)
                    np.savez_compressed(
                            self.output_path / f"{self.iteration}.npz",
                            cell_points = self.cell_points,
                            circumcenters = circumcenters,
                            cell_types = self.cell_types,
                            target_areas = self.target_areas,
                            adjacency = self.adjacency,
                            triangles = self.triangles,
                            boundary_cycle_mask = self.boundary_cycle_mask
                            )

            self.iteration += 1


def calculate_forces(
    num_cells: int,
    max_cells: int,
    max_degree: int,
    spring_length: float,
    critical_delta: float,
    cell_points: np.ndarray,
    cell_types: np.ndarray,
    target_areas: np.ndarray,
    triangles: np.ndarray,
    adjacency: np.ndarray
):
    spring_forces = np.zeros((max_cells, max_degree), dtype=np.float32)
    pressure_forces = np.zeros((max_cells, max_degree), dtype=np.float32)
    unit_vectors = np.zeros((max_cells, max_degree, 2), dtype=np.float32)

    circumcenters = geometry.calculate_circumcenters(cell_points, triangles)

    for i in range(num_cells):
        mask = adjacency[i] != -1
        neighbours = adjacency[i][mask]
        num_neighbours = np.sum(mask)
        if num_neighbours == 0:
            continue

        differences = cell_points[neighbours] - cell_points[i]
        dists = np.sqrt((differences ** 2).sum(axis=1))
        units = differences / dists[:, None]

        unit_vectors[i][mask] = units
        spring_forces[i][mask] = spring_length - dists

        if cell_types[i] != 1:
            area = geometry.calculate_cell_area(i, cell_points[i], triangles, circumcenters)
            split_area_difference = (target_areas[i] - area) / num_neighbours
            pressure_forces[i][mask] += split_area_difference
            neighbour_mask = adjacency[neighbours] == i
            pressure_forces[neighbours][neighbour_mask] += split_area_difference

    # FIXME: This is pretty suspicious
    spring_forces[cell_types == 1] *= 0.1
    full_mask = adjacency != -1
    spring_forces[full_mask] = np.clip(spring_forces[full_mask], -critical_delta, None) 
    forces = (spring_forces + pressure_forces)[..., None] * unit_vectors

    return forces


def generate_filename(x: int, y: int, center_only: bool, random_layout: bool):
    timestamp = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
    center_string = "_center" if center_only else ""
    layout_string = "random" if random_layout else "hexagonal"

    return f"{timestamp}_{x}x{y}{center_string}_{layout_string}"
