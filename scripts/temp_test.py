import matplotlib
import numpy as np
matplotlib.use("Qt5Agg")

from ciliasim import examine

tissue = examine.Examine("./simulations/21-11-26_20-08-34_40x40_random/", "./animations/")
tissue.animate(plot_type="major-axes", save=True)
