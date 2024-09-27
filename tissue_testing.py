from tissue import Tissue
import numpy as np
import cProfile

def main():
    tissue = Tissue(10, 10, 0.06)
    tissue.set_tracking()
    tissue.set_plotting()
    #tissue.set_center_only(True)
    tissue.random_layout()
    tissue.set_plot_spring()
    #tissue.simulate(f"Tissue annealing - No cilia force.", 1000, 100, plotting=False)
    tissue.simulate(f"Tissue annealing - No cilia force.", 1000, 100)
    #tissue.set_uniform_cilia_forces([0, 1], 0.8)
    tissue.set_random_cilia_forces(1.2)
    #tissue.simulate(f"Tissue under cilia force.", 5000, 100, plotting=False)
    tissue.simulate(f"Tissue under cilia force.", 5000, 100)

    tissue.write_to_file("./saved_simulations/test26.json")

#cProfile.run('main()', sort="tottime")
main()
