import matplotlib
import numpy as np
matplotlib.use("Qt5Agg")

from ciliasim import examine

tissue = examine.Examine("./simulations/25-11-26_12-02-13_20x20_random/", "./animations/")
tissue.animate(plot_type="area-strain", save=True)
