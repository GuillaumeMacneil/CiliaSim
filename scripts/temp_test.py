import matplotlib
import numpy as np
matplotlib.use("Qt5Agg")

from ciliasim import examine

tissue = examine.Examine("./simulations/21-11-26_20-32-18_20x20_random/", "./animations/")
tissue.animate(plot_type="spring", save=True)
