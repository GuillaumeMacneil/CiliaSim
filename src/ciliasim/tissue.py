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
        if random_layout:
            self.cell_points = initialisation.uniform_random_layout(self.x, self.y, self.num_cells, self.max_cells)
        else:
            self.cell_points, self.num_cells = initialisation.hexagonal_layout(self.x, self.y, self.num_cells, self.max_cells)

        self.cell_types = np.full(self.max_cells, -1, dtype=np.int8)
        self.target_areas = np.zeros(self.max_cells, dtype=np.float32)
        self.adjacency = np.full((self.max_cells, self.max_degree), -1, dtype=np.int32)
        self.max_triangles = (2 * self.max_cells) - 5
        self.triangles = np.full((self.max_triangles, 3), -1, dtype=np.int32)
        self.cell_triangles = np.full((self.max_cells, self.max_degree), -1, dtype=np.int32)
        # FIXME: Provide a proper default initialisation
        self.boundary_cycle = None
        self.boundary_cycle_mask = np.zeros(self.max_cells, dtype=bool)

        # Force state arrays
        self.cilia_forces = np.zeros((self.max_cells, 2), dtype=np.float32)
        self.flow_force = np.zeros(2, dtype=np.float32)

    def specify_cells(self, area_distribution: tuple[float, float]):
        # Determine the boundary cycle and specify the individual cell types
        self.cell_types[:self.num_cells] = 0
        
        self.boundary_cycle_mask, self.boundary_cycle, self.num_cells = boundary.create_boundary(self.cell_points, self.num_cells, self.max_cells, self.spring_length)

        self.cell_types[self.boundary_cycle] = 1
        self.target_areas[self.boundary_cycle] = 0
        
        self.adjacency, self.triangles, self.cell_triangles = connectivity.full_update(self.num_cells, self.max_cells, self.max_degree, self.max_triangles, self.cell_points, self.boundary_cycle, self.boundary_cycle_mask)

        boundary.constrain_to_cycle(self.adjacency, self.boundary_cycle, self.boundary_cycle_mask, self.max_degree)

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

        self.evaluate_boundary()
        self.validate_initial_geometry()
        
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

    def validate_initial_geometry(self):
        circumcenters = geometry.calculate_circumcenters(self.cell_points, self.triangles)
        areas = np.asarray(geometry.calculate_cell_areas(circumcenters, self.cell_triangles))

        real_mask = (self.cell_types != 1) & (self.cell_types != -1)
        real_indices = np.where(real_mask)[0]
        real_areas = areas[real_mask]
        real_targets = self.target_areas[real_mask]
        relative_areas = real_areas / real_targets

        if not np.all(np.isfinite(relative_areas)):
            bad = real_indices[~np.isfinite(relative_areas)]
            raise RuntimeError(f"Initial geometry contains non-finite Voronoi areas for cells {bad}.")

        min_relative_area = np.min(relative_areas)
        median_relative_area = np.median(relative_areas)
        max_relative_area = np.max(relative_areas)

        print(f"Initial geometry: A/A0 min={min_relative_area:.3f}, median={median_relative_area:.3f}, max={max_relative_area:.3f}")

        if max_relative_area <= 5.0:
            return

        worst_index = real_indices[np.argmax(relative_areas)]

        triangle_indices = self.cell_triangles[worst_index]
        triangle_indices = triangle_indices[triangle_indices != -1]

        local_triangles = self.triangles[triangle_indices]
        local_cells = np.unique(local_triangles)

        print()
        print(f"Pathological initial cell: {worst_index}")
        print(f"Position: {self.cell_points[worst_index]}")
        print(f"Area: {areas[worst_index]}")
        print(f"Target area: {self.target_areas[worst_index]}")
        print(f"A/A0: {areas[worst_index] / self.target_areas[worst_index]}")
        print(f"Adjacency: {self.adjacency[worst_index]}")
        print(f"Incident triangles: {triangle_indices}")

        print()
        print("Local cells:")

        for cell in local_cells:
            print(f"Cell {cell}: position={self.cell_points[cell]}, type={self.cell_types[cell]}, boundary={self.boundary_cycle_mask[cell]}")

        print()
        print("Incident triangles:")

        for triangle_index in triangle_indices:
            triangle = self.triangles[triangle_index]
            points = self.cell_points[triangle]

            a = points[0]
            b = points[1]
            c = points[2]

            ab = b - a
            ac = c - a

            twice_area = np.abs(ab[0] * ac[1] - ab[1] * ac[0])

            edge_ab = np.linalg.norm(b - a)
            edge_bc = np.linalg.norm(c - b)
            edge_ca = np.linalg.norm(a - c)
            max_edge = max(edge_ab, edge_bc, edge_ca)

            quality = twice_area / (max_edge ** 2) if max_edge > 0 else 0.0

            circumcenter = np.asarray(circumcenters[triangle_index])
            circumradius = np.linalg.norm(circumcenter - a)
            radius_ratio = circumradius / max_edge if max_edge > 0 else np.inf

            print(
                f"Triangle {triangle_index}: {triangle}, "
                f"types={self.cell_types[triangle]}, "
                f"boundary={self.boundary_cycle_mask[triangle]}, "
                f"quality={quality:.3e}, "
                f"R/L={radius_ratio:.3e}, "
                f"circumcenter={circumcenter}"
            )

        raise RuntimeError(f"Initial boundary geometry is pathological: cell {worst_index} has A/A0={max_relative_area:.3f}.")

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

    def old_simulate(self, iterations: int):
        for _ in tqdm(range(iterations)):
            # Calculate internal forces
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
            bad_forces = np.where(~np.all(np.isfinite(internal_force), axis=1))[0]
            if len(bad_forces) > 0:
                print("Bad forces:", bad_forces)
                print("Forces:", internal_force[bad_forces])
                print("Positions:", self.cell_points[bad_forces])

            # Introduce external forces and move cell centers accordingly
            multiciliated_mask = self.cell_types == 2
            total_force = internal_force + self.cilia_forces
            total_force[multiciliated_mask] += self.flow_force
            self.cell_points[:self.num_cells] += total_force[:self.num_cells] * self.mobility * self.dt
            self.evaluate_boundary()
            self.validate_initial_geometry()

            bad = np.where(np.all(self.cell_points[:self.num_cells] == -1, axis=1))[0]
            if len(bad) > 0:
                print("[WARNING] Padded in active rows:", bad)
            bad_refs = np.unique(self.adjacency[:self.num_cells][self.adjacency[:self.num_cells] >= self.num_cells])
            if len(bad_refs) > 0 :
                print("[WARNING] References to out-of-range indices:", bad_refs)

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
                        boundary_cycle_mask = self.boundary_cycle_mask
                    )

            self.iteration += 1

    def simulate(self, iterations: int):
        for _ in tqdm(range(iterations)):
            # Calculate internal forces
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

            # Check for non-finite forces
            bad_forces = np.where(~np.all(np.isfinite(internal_force), axis=1))[0]
            if len(bad_forces) > 0:
                print(f"\nNon-finite forces at iteration {self.iteration}")
                print("Cells:", bad_forces)
                print("Forces:", internal_force[bad_forces])
                print("Positions:", self.cell_points[bad_forces])
                raise RuntimeError("Non-finite internal force")

            # Introduce external forces
            multiciliated_mask = self.cell_types == 2
            total_force = internal_force + self.cilia_forces
            total_force[multiciliated_mask] += self.flow_force

            # Calculate proposed displacement without applying it yet
            displacement = total_force[:self.num_cells] * self.mobility * self.dt
            displacement_magnitudes = np.linalg.norm(displacement, axis=1)

            max_displacement_index = np.argmax(displacement_magnitudes)
            max_displacement = displacement_magnitudes[max_displacement_index]

            # Stop before a large displacement corrupts the geometry
            if max_displacement > 0.1 * self.spring_length:
                cell_index = max_displacement_index

                circumcenters = geometry.calculate_circumcenters(self.cell_points, self.triangles)
                areas = np.asarray(geometry.calculate_cell_areas(circumcenters, self.cell_triangles))

                triangle_indices = self.cell_triangles[cell_index]
                triangle_indices = triangle_indices[triangle_indices != -1]

                print(f"\nMechanical instability at iteration {self.iteration}")
                print(f"Cell: {cell_index}")
                print(f"Cell type: {self.cell_types[cell_index]}")
                print(f"Boundary: {self.boundary_cycle_mask[cell_index]}")
                print(f"Position: {self.cell_points[cell_index]}")
                print(f"Area: {areas[cell_index]}")
                print(f"Target area: {self.target_areas[cell_index]}")
                print(f"Internal force: {internal_force[cell_index]}")
                print(f"Cilia force: {self.cilia_forces[cell_index]}")
                print(f"Total force: {total_force[cell_index]}")
                print(f"Displacement: {displacement[cell_index]}")
                print(f"Displacement magnitude: {max_displacement}")
                print(f"Adjacency: {self.adjacency[cell_index]}")
                print(f"Incident triangles: {triangle_indices}")

                for triangle_index in triangle_indices:
                    triangle = self.triangles[triangle_index]
                    points = self.cell_points[triangle]

                    a = points[0]
                    b = points[1]
                    c = points[2]

                    ab = b - a
                    ac = c - a

                    twice_area = np.abs(ab[0] * ac[1] - ab[1] * ac[0])

                    edge_ab = np.linalg.norm(b - a)
                    edge_bc = np.linalg.norm(c - b)
                    edge_ca = np.linalg.norm(a - c)
                    max_edge = max(edge_ab, edge_bc, edge_ca)

                    quality = twice_area / (max_edge ** 2) if max_edge > 0 else 0.0

                    print()
                    print(f"Triangle {triangle_index}: {triangle}")
                    print(f"Points:\n{points}")
                    print(f"Circumcenter: {np.asarray(circumcenters[triangle_index])}")
                    print(f"Quality: {quality:.6e}")

                    print()
                    boundary.debug_boundary_triangle(
                        self.adjacency,
                        self.boundary_cycle_mask,
                        self.cell_points,
                        self.triangles[triangle_index],
                    )

                raise RuntimeError("Mechanical instability detected")

            # Apply movement
            self.cell_points[:self.num_cells] += displacement

            # Update topology/boundary
            self.evaluate_boundary()
            self.validate_initial_geometry()

            # Validate active rows
            bad = np.where(np.all(self.cell_points[:self.num_cells] == -1, axis=1))[0]
            if len(bad) > 0:
                print("[WARNING] Padded in active rows:", bad)

            # Validate adjacency references
            bad_refs = np.unique(
                self.adjacency[:self.num_cells][
                    self.adjacency[:self.num_cells] >= self.num_cells
                ]
            )

            if len(bad_refs) > 0:
                print("[WARNING] References to out-of-range indices:", bad_refs)

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
                        self.cell_points,
                        self.triangles,
                    )

                    np.savez_compressed(
                        self.output_path / f"{self.iteration:06d}.npz",
                        cell_points=self.cell_points,
                        circumcenters=circumcenters,
                        cell_types=self.cell_types,
                        target_areas=self.target_areas,
                        adjacency=self.adjacency,
                        triangles=self.triangles,
                        cell_triangles=self.cell_triangles,
                        boundary_cycle_mask=self.boundary_cycle_mask,
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
