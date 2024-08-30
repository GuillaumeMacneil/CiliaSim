from tissue import Tissue
import numpy as np
import cProfile

def main():
    tissue = Tissue(20, 20, 0.06)
    tissue.set_tracking()
    #tissue.set_plotting()
    tissue.set_center_only(True)
    tissue.hexagonal_grid_layout()
    tissue.set_plot_force_vector_rel()
    #tissue.simulate(f"Tissue annealing - No cilia force.", 1000, 100, plotting=False)
    tissue.simulate(f"Tissue annealing - No cilia force.", 1000, 100)
    tissue.set_uniform_cilia_forces([0, 1], 0.75)
    #tissue.simulate(f"Tissue under cilia force.", 1000, 100, plotting=False)
    tissue.simulate(f"Tissue under cilia force.", 1000, 100)

    tissue.write_to_file("./saved_simulations/test22.json")

#cProfile.run('main()', sort="tottime")
main()
