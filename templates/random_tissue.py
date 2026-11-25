import numpy as np

from ciliasim import tissue


N = 20

simulation = tissue.Tissue(
    x=N,
    y=N,
    density=0.06,
    center_only=False,
    random_layout=True,
    save=True,
    save_freq=10,
    output_dir="simulations/",
)

target_area = np.sqrt(3) / 2 * simulation.spring_length ** 2
simulation.specify_cells((target_area, 0.0))

simulation.simulate(1000)

simulation.set_uniform_ciliary_forces(
    np.array([0.0, 1.0]),
    magnitude=1.0,
)

simulation.simulate(1000)
