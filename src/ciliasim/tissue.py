import json
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from ciliasim import boundary, connectivity, geometry, initialisation


class Tissue:
    """Represent and simulate a two-dimensional Voronoi tissue."""

    def __init__(
        self,
        x: int,
        y: int,
        density: float,
        spring_length: float = 1.0,
        spring_constant: float = 1.0,
        area_constant: float = 1.0,
        perimeter_constant: float = 1.0,
        oversize_factor: float = 3.0,
        max_degree: int = 15,
        center_only: bool = False,
        random_layout: bool = False,
        mobility: float = 0.95,
        dt: float = 0.01,
        save: bool = False,
        save_freq: int = 10,
        output_dir: str = "",
    ):
        """
        Initialize the tissue geometry, mechanics, and simulation state.

        Args:
            x: Width of the initial tissue domain.
            y: Height of the initial tissue domain.
            density: Target fraction of multiciliated cells.
            spring_length: Preferred distance between neighbouring cell centres.
            spring_constant: Stiffness of centre-to-centre springs.
            area_constant: Strength of the cell area energy.
            perimeter_constant: Strength of the cell perimeter energy.
            oversize_factor: Storage multiplier allowing additional boundary cells.
            max_degree: Maximum number of neighbours and incident triangles per cell.
            center_only: Whether to create only one central multiciliated cell.
            random_layout: Whether to use a random rather than hexagonal layout.
            mobility: Cell mobility used to convert force into velocity.
            dt: Simulation timestep.
            save: Whether to save simulation states.
            save_freq: Number of iterations between saved states.
            output_dir: Directory in which simulation output is stored.

        Returns:
            None.
        """
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
        self.perimeter_constant = perimeter_constant

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
            raise ValueError(
                "`save`, `save_freq` and `output_dir` must be provided for state saving."
            )
        else:
            self.output_path = Path(output_dir) / generate_filename(
                x, y, center_only, random_layout
            )
            self.output_path.mkdir(parents=True, exist_ok=False)

        # Positional state arrays
        self.cell_points = np.full((self.max_cells, 2), -1, np.float32)

        if self.random_layout:
            self.cell_points = initialisation.uniform_random_layout(
                self.x,
                self.y,
                self.num_cells,
                self.max_cells,
                self.spring_length,
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
        self.cell_triangles = np.full(
            (self.max_cells, self.max_degree), -1, dtype=np.int32
        )
        self.boundary_cycle = np.zeros(self.max_cells, dtype=np.int32)
        self.boundary_cycle_mask = np.zeros(self.max_cells, dtype=bool)

        # Force state arrays
        self.cilia_forces = np.zeros((self.max_cells, 2), dtype=np.float32)
        self.flow_force = np.zeros(2, dtype=np.float32)

    def specify_cells(self, area_distribution: tuple[float, float]):
        """
        Construct the boundary and assign cell types and target areas.

        Args:
            area_distribution: Mean and standard deviation of target cell areas.
                A zero standard deviation produces uniform target areas.

        Returns:
            None.
        """
        # Determine the boundary cycle and specify the individual cell types
        self.cell_types[: self.num_cells] = 0

        if self.random_layout:
            self.boundary_cycle_mask, self.boundary_cycle, self.num_cells = (
                boundary.create_boundary(
                    self.cell_points,
                    self.num_cells,
                    self.max_cells,
                    self.spring_length,
                )
            )
        else:
            self.boundary_cycle_mask, self.boundary_cycle, self.num_cells = (
                boundary.create_hexagonal_boundary(
                    self.cell_points,
                    self.num_cells,
                    self.max_cells,
                    self.spring_length,
                )
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

        boundary.constrain_to_cycle(
            self.adjacency,
            self.boundary_cycle,
            self.boundary_cycle_mask,
            self.max_degree,
        )

        if self.center_only:
            area_center = np.array([self.x / 2, self.y / 2])
            real_indices = np.where(self.cell_types == 0)[0]
            center_distances = np.sum(
                (self.cell_points[real_indices] - area_center) ** 2, axis=1
            )
            center_cell = real_indices[np.argmin(center_distances)]
            self.cell_types[center_cell] = 2
        else:
            current_density = 0
            while current_density < self.density:
                candidates = np.where(self.cell_types[: self.num_cells] == 0)[0]
                if candidates.size == 0:
                    break

                chosen_cell = np.random.choice(candidates)
                self.cell_types[chosen_cell] = 2
                neighbours = self.adjacency[chosen_cell]
                neighbours = neighbours[neighbours >= 0]

                for cell_index in neighbours:
                    if self.cell_types[cell_index] == 0:
                        self.cell_types[cell_index] = 3

                current_density = (
                    len(np.where(self.cell_types == 2)[0]) / self.num_cells
                )

            self.cell_types[np.where(self.cell_types == 3)[0]] = 0

        # Assign each cell a target area
        if area_distribution[1] == 0:
            self.target_areas = initialisation.uniform_target_areas(
                self.cell_types, area_distribution[0]
            )
        else:
            self.target_areas = initialisation.normal_random_target_areas(
                self.cell_types, area_distribution[0], area_distribution[1]
            )

    def set_uniform_ciliary_forces(self, direction: np.ndarray, magnitude: float):
        """
        Assign the same ciliary force to every multiciliated cell.

        Args:
            direction: Direction vector of the applied ciliary force.
            magnitude: Scalar multiplier applied to the direction vector.

        Returns:
            None.
        """
        force = direction * magnitude
        multiciliated_cells = np.where(self.cell_types == 2)[0]
        self.cilia_forces[multiciliated_cells] = force

    def set_random_ciliary_forces(self, magnitude: float):
        """
        Assign randomly oriented ciliary forces to multiciliated cells.

        Args:
            magnitude: Magnitude of each ciliary force.

        Returns:
            None.
        """
        multiciliated_cells = np.where(self.cell_types == 2)[0]
        non_unit_directions = np.random.uniform(-1, 1, [len(multiciliated_cells), 2])
        unit_directions = non_unit_directions / np.linalg.norm(
            non_unit_directions, axis=1, keepdims=True
        )
        self.cilia_forces[multiciliated_cells] = unit_directions * magnitude

    def set_flow_force(self, direction: np.ndarray, magnitude: float):
        """
        Set the global flow force applied to multiciliated cells.

        Args:
            direction: Direction vector of the flow force.
            magnitude: Scalar multiplier applied to the direction vector.

        Returns:
            None.
        """
        self.flow_force = direction * magnitude

    def evaluate_boundary(self):
        """
        Adapt the ghost-cell boundary and rebuild tissue connectivity.

        Args:
            None.

        Returns:
            None.
        """
        max_iterations = 20

        # Repeatedly add and remove ghost cells until the boundary stabilizes.
        for _ in range(max_iterations):
            prev_num_cells = self.num_cells

            self.num_cells, self.boundary_cycle = (
                boundary.add_additional_boundary_cells(
                    self.adjacency,
                    self.boundary_cycle,
                    self.boundary_cycle_mask,
                    self.cell_points,
                    self.cell_types,
                    self.target_areas,
                )
            )
            changed = self.num_cells != prev_num_cells
            self.num_cells, self.boundary_cycle = (
                boundary.remove_excessive_boundary_cells(
                    self.adjacency,
                    self.boundary_cycle,
                    self.boundary_cycle_mask,
                    self.cell_points,
                    self.cell_types,
                    self.target_areas,
                )
            )
            changed = (self.num_cells != prev_num_cells) or changed

            if not changed:
                break

            # Compact active cells into the leading rows required by full_update().
            empty_mask = np.all(self.cell_points == -1, axis=1)
            new_order = np.concatenate(
                [np.nonzero(~empty_mask)[0], np.nonzero(empty_mask)[0]]
            )

            # Remap the boundary indices to the compacted cell ordering.
            inverse_order = np.empty_like(new_order)
            inverse_order[new_order] = np.arange(len(new_order))
            self.boundary_cycle = inverse_order[self.boundary_cycle]

            self.cell_points = self.cell_points[new_order]
            self.boundary_cycle_mask = self.boundary_cycle_mask[new_order]
            self.cell_types = self.cell_types[new_order]
            self.target_areas = self.target_areas[new_order]

            self.adjacency, self.triangles, self.cell_triangles = (
                connectivity.full_update(
                    self.num_cells,
                    self.max_cells,
                    self.max_degree,
                    self.max_triangles,
                    self.cell_points,
                    self.boundary_cycle,
                    self.boundary_cycle_mask,
                )
            )
        else:
            raise RuntimeError("Boundary failed to converge.")

        if not changed:
            # FIXME: This should correctly implement partial updating
            self.adjacency, self.triangles, self.cell_triangles = (
                connectivity.full_update(
                    self.num_cells,
                    self.max_cells,
                    self.max_degree,
                    self.max_triangles,
                    self.cell_points,
                    self.boundary_cycle,
                    self.boundary_cycle_mask,
                )
            )

        boundary.constrain_to_cycle(
            self.adjacency,
            self.boundary_cycle,
            self.boundary_cycle_mask,
            self.max_degree,
        )

    def simulate(self, iterations: int):
        """
        Advance the tissue through a specified number of simulation iterations.

        Args:
            iterations: Number of timesteps to simulate.

        Returns:
            None.
        """
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
                self.perimeter_constant,
            )

            internal_force = np.asarray(internal_force)

            # External forces
            total_force = internal_force + self.cilia_forces
            multiciliated_mask = self.cell_types == 2
            total_force[multiciliated_mask] += self.flow_force

            real_mask = (self.cell_types[: self.num_cells] != 1) & (
                self.cell_types[: self.num_cells] != -1
            )
            force_magnitudes = np.linalg.norm(
                internal_force[: self.num_cells][real_mask], axis=1
            )

            displacement = total_force[: self.num_cells] * self.mobility * self.dt
            displacement_magnitudes = np.linalg.norm(displacement[real_mask], axis=1)

            self.cell_points[: self.num_cells] += displacement
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
                    circumcenters = geometry.calculate_circumcenters(
                        self.cell_points, self.triangles
                    )
                    np.savez_compressed(
                        self.output_path / f"{self.iteration:06d}.npz",
                        iteration=self.iteration,
                        cell_points=self.cell_points,
                        circumcenters=circumcenters,
                        cell_types=self.cell_types,
                        target_areas=self.target_areas,
                        adjacency=self.adjacency,
                        triangles=self.triangles,
                        cell_triangles=self.cell_triangles,
                        boundary_cycle_mask=self.boundary_cycle_mask,
                        boundary_cycle=self.boundary_cycle,
                        internal_force=internal_force,
                        cilia_force=self.cilia_forces,
                        flow_force=self.flow_force,
                        total_force=total_force,
                    )

            self.iteration += 1


def calculate_energy(
    cell_points,
    cell_types,
    target_areas,
    triangles,
    cell_triangles,
    adjacency,
    spring_length,
    spring_constant,
    area_constant,
    perimeter_constant,
):
    """
    Calculate the total mechanical energy of the tissue.

    Args:
        cell_points: Positions of all cell centres.
        cell_types: Cell type array.
        target_areas: Preferred area of each cell.
        triangles: Padded Delaunay triangle array.
        cell_triangles: Ordered incident-triangle indices for each cell.
        adjacency: Padded cell adjacency array.
        spring_length: Preferred distance between neighbouring cell centres.
        spring_constant: Strength of the spring energy.
        area_constant: Strength of the area energy.
        perimeter_constant: Strength of the perimeter energy.

    Returns:
        Total spring, area, and perimeter energy of the tissue.
    """
    valid_neighbours = adjacency != -1
    safe_adjacency = jnp.where(valid_neighbours, adjacency, 0)

    neighbour_points = cell_points[safe_adjacency]
    differences = neighbour_points - cell_points[:, None, :]

    safe_differences = jnp.where(
        valid_neighbours[..., None], differences, jnp.array([1.0, 0.0])
    )
    distances = jnp.linalg.norm(safe_differences, axis=-1)

    spring_extensions = distances - spring_length
    spring_energy = (
        0.25
        * spring_constant
        * jnp.sum(jnp.where(valid_neighbours, spring_extensions**2, 0.0))
    )

    circumcenters = geometry.calculate_circumcenters(cell_points, triangles)

    areas = geometry.calculate_cell_areas(circumcenters, cell_triangles)
    perimeters = geometry.calculate_cell_perimeters(circumcenters, cell_triangles)

    real_cells = (cell_types != 1) & (cell_types != -1)

    area_errors = areas - target_areas
    area_energy = (
        0.5 * area_constant * jnp.sum(jnp.where(real_cells, area_errors**2, 0.0))
    )

    target_perimeter = 2.0 * jnp.sqrt(3.0) * spring_length
    perimeter_errors = perimeters - target_perimeter
    perimeter_energy = (
        0.5
        * perimeter_constant
        * jnp.sum(jnp.where(real_cells, perimeter_errors**2, 0.0))
    )

    return spring_energy + area_energy + perimeter_energy


# Differentiate total energy with respect to cell-centre positions.
calculate_forces = jax.jit(jax.grad(calculate_energy, argnums=0))


def generate_filename(x: int, y: int, center_only: bool, random_layout: bool):
    """
    Generate a timestamped directory name describing a simulation.

    Args:
        x: Width of the initial tissue domain.
        y: Height of the initial tissue domain.
        center_only: Whether the tissue contains only one central MCC.
        random_layout: Whether the tissue uses a random initial layout.

    Returns:
        Timestamped simulation name containing the main layout parameters.
    """
    timestamp = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
    center_string = "_center" if center_only else ""
    layout_string = "random" if random_layout else "hexagonal"

    return f"{timestamp}_{x}x{y}{center_string}_{layout_string}"
