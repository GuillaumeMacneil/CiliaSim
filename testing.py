from tissue import Tissue
import numpy as np
import cProfile

def main():
    tissue = Tissue(20, 20, 0.1)
    tissue.set_center_only(True)
    tissue.random_layout()
    tissue.anneal("Tissue annealing - spring tension only.")
    tissue.simulate("Tissue annealing - tension and pressure.", 1000)
    tissue.set_flow(np.array([0, 1]), 0.5)
    tissue.set_plot_major_axes()
    tissue.simulate("Tissue dynamics with cilia force.", 10000)

#cProfile.run('main()', sort="tottime")
main()
