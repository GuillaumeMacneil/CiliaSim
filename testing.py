from tissue import Tissue
import numpy as np
import cProfile

def main():
    tissue = Tissue(10, 10, 0.1)
    tissue.set_tracking()
    tissue.set_center_only(True)
    tissue.random_layout()
    tissue.set_plot_basic()
    tissue.simulate(f"Tissue annealing.", 1000, 100)
    tissue.set_flow([0, 1], 1)
    tissue.simulate("Tissue dynamics with cilia force.", 4000, 1000)
    tissue.write_to_file("./saved_simulations/test2.json")

#cProfile.run('main()', sort="tottime")
main()
