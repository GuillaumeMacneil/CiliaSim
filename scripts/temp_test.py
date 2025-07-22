import matplotlib
import numpy as np
matplotlib.use("Qt5Agg")

from ciliasim import examine

tissue = examine.Examine("./simulations/22-07-25_18-58-14_10x10_center_hexagonal/", "./animations/")
tissue.animate(plot_type="spring", save=True)
