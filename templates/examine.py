import matplotlib
from pathlib import Path
from datetime import datetime

matplotlib.use("Qt5Agg")

from ciliasim import examine


simulation_dir = None
simulation_root = Path("simulations/")


if simulation_dir is None:
    simulation_dirs = [path for path in simulation_root.iterdir() if path.is_dir()]

    if not simulation_dirs:
        raise RuntimeError("No saved simulations found.")

    def get_simulation_time(path):
        try:
            return datetime.strptime(path.name[:17], "%d-%m-%y_%H-%M-%S").timestamp()
        except ValueError:
            return path.stat().st_mtime

    simulation_dir = max(simulation_dirs, key=get_simulation_time)


simulation = examine.Examine(
    simulation_dir,
    "animations/",
)

simulation.animate(
    plot_type="major-axes",
    save=True,
)
