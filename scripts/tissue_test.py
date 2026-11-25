import matplotlib
import numpy as np
matplotlib.use("Qt5Agg")

from ciliasim import tissue

tissue = tissue.Tissue(
            x = 20, 
            y = 20, 
            density = 0.06,
            center_only=True,
            random_layout=False,
            save=True,
            save_freq=10,
            output_dir="simulations/"
        )

tissue.specify_cells((0.866, 0.0))
tissue.simulate(1000)
tissue.set_uniform_ciliary_forces(np.array([0, 0.75]), 1)
tissue.simulate(2000)
