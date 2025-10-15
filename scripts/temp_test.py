import matplotlib
import numpy as np
matplotlib.use("Qt5Agg")

from ciliasim import examine

tissue = examine.Examine("./simulations/15-10-25_22-08-13_10x10_center_random/", "./animations/")
tissue.animate(plot_type="spring", save=True)
