import matplotlib
import numpy as np
matplotlib.use("Qt5Agg")

from ciliasim import examine

tissue = examine.Examine("./simulations/25-11-26_14-23-26_20x20_center_hexagonal/", "./animations/")
tissue.animate(plot_type="major-axes", save=True)
