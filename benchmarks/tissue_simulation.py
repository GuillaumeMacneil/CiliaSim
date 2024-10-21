from ciliasim import Tissue
import argparse
from typing import cast
from time import time_ns

parser = argparse.ArgumentParser()

parser.add_argument("--tissue_size", type=str, default="15x15")
parser.add_argument("--cilia_density", type=float, default=0.06)
parser.add_argument("--annealing_steps", type=int, default=1000)
parser.add_argument("--simulation_steps", type=int, default=5000)


def run_benchmark(
    tissue_size: tuple[int, int],
    cilia_density: float,
    annealing_steps: int = 1000,
    simulation_steps: int = 5000,
):
    tissue = Tissue(*tissue_size, cilia_density)

    tissue.set_tracking()
    tissue.set_center_only(True)
    tissue.hexagonal_grid_layout()

    annealing_start = time_ns()

    tissue.simulate(
        f"Tissue annealing - No cilia force.",
        annealing_steps,
        100,
        plotting=False,
        progress=True,
    )

    annealing_end = time_ns()
    time_per_step_ms = (annealing_end - annealing_start) / annealing_steps / 1e6

    print(f"Annealing time per step: {time_per_step_ms:.2f} ms")

    sim_start = time_ns()

    tissue.set_uniform_cilia_forces([0, 1], 0.5)
    tissue.simulate(
        f"Tissue under random cilia force of mag. {0.5}.",
        simulation_steps,
        100,
        plotting=False,
        progress=True,
    )

    sim_end = time_ns()
    time_per_step_ms = (sim_end - sim_start) / simulation_steps / 1e6

    print(f"Simulation time per step: {time_per_step_ms:.2f} ms")


if __name__ == "__main__":
    args = parser.parse_args()
    tissue_size = tuple(map(int, args.tissue_size.split("x")))
    run_benchmark(
        cast(tuple[int, int], tissue_size),
        args.cilia_density,
        args.annealing_steps,
        args.simulation_steps,
    )
