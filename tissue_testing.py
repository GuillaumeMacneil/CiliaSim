from tissue import Tissue
import numpy as np
import cProfile

def main():
    tissue = Tissue(15, 15, 0.06)
    tissue.set_tracking()
    tissue.set_center_only(True)
    tissue.hexagonal_grid_layout()
    tissue.set_plot_force_vector_abs()
    tissue.simulate(f"Tissue annealing - No cilia force.", 2000, 100)
    tissue.set_uniform_cilia_forces([0, 1], 0.5)
    tissue.simulate(f"Tissue under cilia force.", 2000, 100)
    tissue.set_flow([0, -1], 0.5)
    tissue.simulate(f"Tissue under cilia force and flow force.", 2000, 100)
    tissue.set_flow([0, 0], 0)
    tissue.set_uniform_cilia_forces([0, 0], 0)
    tissue.simulate(f"Tissue under no force.", 2000, 100)

    tissue.write_to_file("./saved_simulations/test18.json")

#cProfile.run('main()', sort="tottime")
main()
