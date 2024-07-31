from tissue import Tissue
import numpy as np
import cProfile

def main():
    tissue = Tissue(15, 15, 0.1)
    tissue.set_tracking()
    tissue.set_center_only(True)
    tissue.random_layout()
    tissue.set_plot_spring()
    tissue.simulate(f"Tissue annealing.", 3000, 100)
    tissue.set_flow([0, 1], 0.5)
    tissue.simulate("Tissue dynamics with cilia force (UP).", 3000, 100)
    tissue.set_flow([1, 0], 0.5)
    tissue.simulate("Tissue dynamics with cilia force (RIGHT).", 3000, 100)
    tissue.set_flow([0, 0], 0)
    tissue.simulate("Tissue dynamics with cilia force.", 1000, 100)
    tissue.write_to_file("./saved_simulations/test16.json")

#cProfile.run('main()', sort="tottime")
main()
