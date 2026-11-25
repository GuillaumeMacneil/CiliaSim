import numpy as np
import jax
import jax.numpy as jnp
import json
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from ciliasim import initialisation, geometry, boundary, connectivity

class Tissue():
    def __init__(
            self, 
            x: int, 
            y: int, 
            density: float,
            spring_length: float = 1.0,
            spring_constant: float = 1.0,
            area_constant: float = 1.0,
            oversize_factor: float = 3.0,
            max_degree: int = 15,
            center_only: bool = False,
            random_layout: bool = False,
            mobility: float = 0.95,
            dt: float = 0.01,
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
        self.random_layout = random_layout
       
        if (x < 2) or (y < 2):
            raise ValueError("Tissue dimensions must be 2x2 or larger.")

        # Mechanical parameters
        self.spring_length = spring_length
        self.spring_constant = spring_constant
        self.area_constant = area_constant
 
        # Composition parameters
        self.density = density
        self.center_only = center_only

        # Simulation parameters
        self.mobility = mobility
        self.dt = dt

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

        if self.random_layout:
            self.cell_points = initialisation.uniform_random_layout(
                self.x,
                self.y,
                self.num_cells,
                self.max_cells,
            )
        else:
            self.cell_points, self.num_cells = initialisation.hexagonal_layout(
                self.x,
                self.y,
                self.num_cells,
                self.max_cells,
                self.spring_length,
            )

        self.cell_types = np.full(self.max_cells, -1, dtype=np.int8)
        self.target_areas = np.zeros(self.max_cells, dtype=np.float32)
        self.adjacency = np.full((self.max_cells, self.max_degree), -1, dtype=np.int32)
        self.max_triangles = (2 * self.max_cells) - 5
        self.triangles = np.full((self.max_triangles, 3), -1, dtype=np.int32)
        self.cell_triangles = np.full((self.max_cells, self.max_degree), -1, dtype=np.int32)
        self.boundary_cycle = np.zeros(self.max_cells, dtype=np.int32)
        self.boundary_cycle_mask = np.zeros(self.max_cells, dtype=bool)

        # Force state arrays
        self.cilia_forces = np.zeros((self.max_cells, 2), dtype=np.float32)
        self.flow_force = np.zeros(2, dtype=np.float32)

    def specify_cells(self, area_distribution: tuple[float, float]):
        # Determine the boundary cycle and specify the individual cell types
        self.cell_types[:self.num_cells] = 0

        if self.random_layout:
            self.boundary_cycle_mask, self.boundary_cycle, self.num_cells = boundary.create_boundary(
                self.cell_points,
                self.num_cells,
                self.max_cells,
                self.spring_length,
            )
        else:
            self.boundary_cycle_mask, self.boundary_cycle, self.num_cells = boundary.create_hexagonal_boundary(
                self.cell_points,
                self.num_cells,
                self.max_cells,
                self.spring_length,
            )

        self.cell_types[self.boundary_cycle] = 1
        self.target_areas[self.boundary_cycle] = 0

        self.adjacency, self.triangles, self.cell_triangles = connectivity.full_update(
            self.num_cells,
            self.max_cells,
            self.max_degree,
            self.max_triangles,
            self.cell_points,
            self.boundary_cycle,
            self.boundary_cycle_mask,
        )

        boundary.constrain_to_cycle(self.adjacency, self.boundary_cycle, self.boundary_cycle_mask, self.max_degree)

        if self.center_only:
            area_center = np.array([self.x / 2, self.y / 2])
            real_indices = np.where(self.cell_types == 0)[0]
            center_distances = np.sum((self.cell_points[real_indices] - area_center) ** 2, axis=1)
            center_cell = real_indices[np.argmin(center_distances)]
            self.cell_types[center_cell] = 2
        else:
            current_density = 0
            while current_density < self.density:
                candidates = np.where(self.cell_types[:self.num_cells] == 0)[0]
                if candidates.size  == 0:
                    break

                chosen_cell = np.random.choice(candidates)
                self.cell_types[chosen_cell] = 2
                neighbours = self.adjacency[chosen_cell]
                neighbours = neighbours[neighbours >= 0]

                for cell_index in neighbours:
                    if self.cell_types[cell_index] == 0:
                        self.cell_types[cell_index] = 3

                current_density = len(np.where(self.cell_types == 2)[0]) / self.num_cells

            self.cell_types[np.where(self.cell_types == 3)[0]] = 0
           
        # Assign each cell a target area
        if area_distribution[1] == 0:
            self.target_areas = initialisation.uniform_target_areas(self.cell_types, area_distribution[0])
        else:
            self.target_areas = initialisation.normal_random_target_areas(self.cell_types, self.num_cells, area_distribution[0], area_distribution[1])

        #self.evaluate_boundary()
        
    def set_uniform_ciliary_forces(self, direction: np.ndarray, magnitude: float):
        force = direction * magnitude
        multiciliated_cells = np.where(self.cell_types == 2)[0]
        self.cilia_forces[multiciliated_cells] = force

    def set_random_ciliary_forces(self, magnitude: float):
        multiciliated_cells = np.where(self.cell_types == 2)[0]
        non_unit_directions = np.random.uniform(-1, 1, [len(multiciliated_cells), 2])
        unit_directions = non_unit_directions / np.linalg.norm(non_unit_directions, axis=1, keepdims=True)
        self.cilia_forces[multiciliated_cells] = unit_directions * magnitude

    def set_flow_force(self, direction: np.ndarray, magnitude: float):
        self.flow_force = direction * magnitude

    def evaluate_boundary(self):
        max_iterations = 20

        for _ in range(max_iterations):
            prev_num_cells = self.num_cells

            self.num_cells, self.boundary_cycle = boundary.add_additional_boundary_cells(self.adjacency, self.boundary_cycle, self.boundary_cycle_mask, self.cell_points, self.cell_types, self.target_areas)
            changed = self.num_cells != prev_num_cells
            self.num_cells, self.boundary_cycle = boundary.remove_excessive_boundary_cells(self.adjacency, self.boundary_cycle, self.boundary_cycle_mask, self.cell_points, self.cell_types, self.target_areas)
            changed = (self.num_cells != prev_num_cells) or changed

            if not changed:
                break

            empty_mask = np.all(self.cell_points == -1, axis=1)
            new_order = np.concatenate([np.nonzero(~empty_mask)[0], np.nonzero(empty_mask)[0]])

            inverse_order = np.empty_like(new_order)
            inverse_order[new_order] = np.arange(len(new_order))
            self.boundary_cycle = inverse_order[self.boundary_cycle]

            self.cell_points = self.cell_points[new_order]
            self.boundary_cycle_mask = self.boundary_cycle_mask[new_order]
            self.cell_types = self.cell_types[new_order]
            self.target_areas = self.target_areas[new_order]

            self.adjacency, self.triangles, self.cell_triangles = connectivity.full_update(self.num_cells, self.max_cells, self.max_degree, self.max_triangles, self.cell_points, self.boundary_cycle, self.boundary_cycle_mask)
        else:
            raise RuntimeError("Boundary failed to converge.")

        if not changed:
            # FIXME: This should correctly implement partial updating
            self.adjacency, self.triangles, self.cell_triangles = connectivity.full_update(self.num_cells, self.max_cells, self.max_degree, self.max_triangles, self.cell_points, self.boundary_cycle, self.boundary_cycle_mask)

        boundary.constrain_to_cycle(self.adjacency, self.boundary_cycle, self.boundary_cycle_mask, self.max_degree)

    def simulate(self, iterations: int):
        progress = tqdm(range(iterations))

        for _ in progress:
            internal_force = -calculate_forces(
                self.cell_points,
                self.cell_types,
                self.target_areas,
                self.triangles,
                self.cell_triangles,
                self.adjacency,
                self.spring_length,
                self.spring_constant,
                self.area_constant,
            )

            internal_force = np.asarray(internal_force)

            # External forces
            total_force = internal_force + self.cilia_forces
            multiciliated_mask = self.cell_types == 2
            total_force[multiciliated_mask] += self.flow_force
            
            real_mask = (self.cell_types[:self.num_cells] != 1) & (self.cell_types[:self.num_cells] != -1)
            force_magnitudes = np.linalg.norm(internal_force[:self.num_cells][real_mask], axis=1)

            displacement = total_force[:self.num_cells] * self.mobility * self.dt
            displacement_magnitudes = np.linalg.norm(displacement[real_mask], axis=1)

            self.cell_points[:self.num_cells] += displacement
            self.evaluate_boundary()

            progress.set_postfix(
                Fmean=f"{np.mean(force_magnitudes):.3e}",
                Fmax=f"{np.max(force_magnitudes):.3e}",
                Dmax=f"{np.max(displacement_magnitudes):.3e}",
                N=self.num_cells,
                Nb=np.sum(self.boundary_cycle_mask),
            )

            if self.save:
                # Add static parameters header
                if self.iteration == 0:
                    with open(self.output_path / "params.json", "w") as params_file:
                        static_params = {
                            "x": self.x,
                            "y": self.y,
                            "density": self.density,
                            "spring_length": self.spring_length,
                            "spring_constant": self.spring_constant,
                            "area_constant": self.area_constant,
                            "max_degree": self.max_degree,
                            "max_cells": self.max_cells,
                            "max_triangles": self.max_triangles,
                            "mobility": self.mobility,
                            "dt": self.dt,
                        }
                        json.dump(static_params, params_file)

                # Save state to file with given frequency
                if self.iteration % self.save_freq == 0:
                    circumcenters = geometry.calculate_circumcenters(self.cell_points, self.triangles)
                    np.savez_compressed(
                        self.output_path / f"{self.iteration:06d}.npz",
                        cell_points = self.cell_points,
                        circumcenters = circumcenters,
                        cell_types = self.cell_types,
                        target_areas = self.target_areas,
                        adjacency = self.adjacency,
                        triangles = self.triangles,
                        cell_triangles = self.cell_triangles,
                        boundary_cycle_mask = self.boundary_cycle_mask,
                        boundary_cycle = self.boundary_cycle,
                        internal_force = internal_force,
                        total_force = total_force,
                    )

            self.iteration += 1


def calculate_energy(cell_points, cell_types, target_areas, triangles, cell_triangles, adjacency, spring_length, spring_constant, area_constant):
    valid_neighbours = adjacency != -1
    safe_adjacency = jnp.where(valid_neighbours, adjacency, 0)

    neighbour_points = cell_points[safe_adjacency]
    differences = neighbour_points - cell_points[:, None, :]

    safe_differences = jnp.where(valid_neighbours[..., None], differences, jnp.array([1.0, 0.0]))
    distances = jnp.linalg.norm(safe_differences, axis=-1)

    spring_extensions = distances - spring_length
    spring_energy = 0.25 * spring_constant * jnp.sum(jnp.where(valid_neighbours, spring_extensions ** 2, 0.0))

    circumcenters = geometry.calculate_circumcenters(cell_points, triangles)
    areas = geometry.calculate_cell_areas(circumcenters, cell_triangles)

    real_cells = (cell_types != 1) & (cell_types != -1)
    area_errors = areas - target_areas
    area_energy = 0.5 * area_constant * jnp.sum(jnp.where(real_cells, area_errors ** 2, 0.0))

    return spring_energy + area_energy


calculate_forces = jax.jit(jax.grad(calculate_energy, argnums=0))


def generate_filename(x: int, y: int, center_only: bool, random_layout: bool):
    timestamp = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
    center_string = "_center" if center_only else ""
    layout_string = "random" if random_layout else "hexagonal"

    return f"{timestamp}_{x}x{y}{center_string}_{layout_string}"
