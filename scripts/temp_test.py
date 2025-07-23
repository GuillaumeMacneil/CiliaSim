import matplotlib
import numpy as np
matplotlib.use("Qt5Agg")

from ciliasim import examine

tissue = examine.Examine("./simulations/23-07-25_10-11-23_10x10_center_hexagonal/", "./animations/")
tissue.animate(plot_type="spring", save=True)
