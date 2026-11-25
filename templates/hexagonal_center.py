import numpy as np

from ciliasim import tissue


N = 20

simulation = tissue.Tissue(
    x=N,
    y=N,
    density=0.0,
    center_only=True,
    random_layout=False,
    save=True,
    save_freq=10,
    output_dir="simulations/",
)

target_area = np.sqrt(3) / 2 * simulation.spring_length ** 2
simulation.specify_cells((target_area, 0.0))
