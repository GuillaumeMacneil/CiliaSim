import matplotlib
import numpy as np
matplotlib.use("Qt5Agg")

from ciliasim import examine

tissue = examine.Examine("./simulations/26-10-25_17-57-00_20x20_center_random/", "./animations/")
tissue.animate(plot_type="spring", save=True)
