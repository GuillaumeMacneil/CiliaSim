import matplotlib
import numpy as np
matplotlib.use("Qt5Agg")

from ciliasim import examine

tissue = examine.Examine("./simulations/27-11-25_16-41-47_30x30_center_hexagonal/", "./animations/")
tissue.animate(plot_type="major-axes", save=True)
