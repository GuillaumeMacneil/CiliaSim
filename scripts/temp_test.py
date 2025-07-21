import matplotlib
import numpy as np
matplotlib.use("Qt5Agg")

from ciliasim import examine

tissue = examine.Examine("./simulations/21-07-25_11-55-10_10x10_center_hexagonal/", "./animations/")
tissue.animate(save=True)
