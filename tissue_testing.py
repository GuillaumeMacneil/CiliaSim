from tissue import Tissue
import numpy as np
import cProfile

def main():
    tissue = Tissue(15, 15, 0.06)
    tissue.set_tracking()
    tissue.set_plotting()
#    tissue.set_center_only(True)
    tissue.random_layout()
    tissue.set_plot_major_axes()
    #tissue.simulate(f"Tissue annealing - No cilia force.", 1000, 100, plotting=False)
    tissue.simulate(f"Tissue annealing - No cilia force.", 1000, 100)
    #tissue.simulate(f"Tissue under cilia force.", 5000, 100, plotting=False)
    for i in range(4, 10):
        tissue.set_uniform_cilia_forces([0, 1], (i * 0.1))
        tissue.simulate(f"Tissue under upward cilia force of mag. {i * 0.1}.", 2000, 100)

    tissue.write_to_file("./saved_simulations/test29.json")

#cProfile.run('main()', sort="tottime")
main()
