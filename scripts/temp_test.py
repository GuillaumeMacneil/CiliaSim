import matplotlib
import numpy as np
matplotlib.use("Qt5Agg")

from ciliasim import examine

tissue = examine.Examine("./simulations/23-11-26_11-06-44_30x10_random/", "./animations/")
tissue.animate(plot_type="major-axes", save=True)
