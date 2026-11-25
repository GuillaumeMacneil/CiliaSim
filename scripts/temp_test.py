import matplotlib
import numpy as np
matplotlib.use("Qt5Agg")

from ciliasim import examine

tissue = examine.Examine("./simulations/25-11-26_11-05-15_20x20_center_hexagonal/", "./animations/")
tissue.animate(plot_type="area-strain", save=True)
