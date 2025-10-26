import matplotlib
matplotlib.use("Qt5Agg")

from ciliasim import geometry
from ciliasim import tissue

tissue = tissue.Tissue(
            x = 20, 
            y = 20, 
            density = 0.06,
            center_only=True,
            random_layout=True,
            save=True,
            save_freq=10,
            output_dir="simulations/"
        )

tissue.specify_cells((1.0, 0.0))
#tissue.set_uniform_ciliary_forces(np.array([0, 1]), 1)
tissue.simulate(2000)
