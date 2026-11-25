import matplotlib
import numpy as np
matplotlib.use("Qt5Agg")

from ciliasim import examine

tissue = examine.Examine("./simulations/25-11-26_10-21-08_30x20_random/", "./animations/")
tissue.animate(plot_type="area-strain", save=True)
