import matplotlib
import numpy as np
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt

from ciliasim import geometry
from ciliasim import tissue

tissue = tissue.Tissue(
            x = 10, 
            y = 10, 
            density = 0.06,
            center_only=True,
            save=True,
            save_freq=1,
            output_dir="simulations/"
        )

tissue.specify_cells((1.0, 0.0))
tissue.set_uniform_ciliary_forces(np.array([0, 1]), 1)
tissue.simulate(50)
