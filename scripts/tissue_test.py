import matplotlib
import numpy as np
matplotlib.use("Qt5Agg")

from ciliasim import tissue

tissue = tissue.Tissue(
            x = 20, 
            y = 20, 
            density = 0.06,
            center_only=False,
            random_layout=True,
            save=True,
            save_freq=10,
            output_dir="simulations/"
        )

target_area = np.sqrt(3) / 2 * tissue.spring_length ** 2
tissue.specify_cells((target_area, 0.0))
tissue.simulate(2000)
tissue.set_uniform_ciliary_forces(np.array([0, 0.75]), 1)
tissue.simulate(2000)
