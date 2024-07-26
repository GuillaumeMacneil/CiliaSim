from tissue import Tissue
import numpy as np
import cProfile

def main():
    tissue = Tissue(10, 10, 0.1)
    tissue.set_tracking()
    tissue.set_center_only(True)
    tissue.hexagonal_grid_layout()
    tissue.set_plot_force_vector()
    tissue.simulate(f"Tissue annealing.", 2000, 100)
    tissue.set_flow([1, 0], 1)
    tissue.simulate("Tissue dynamics with cilia force.", 4000, 100)
    tissue.set_flow([0, 0], 0)
    tissue.simulate("Tissue dynamics with cilia force.", 2000, 100)
    tissue.write_to_file("./saved_simulations/test11.json")

#cProfile.run('main()', sort="tottime")
main()
