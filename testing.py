from tissue import Tissue
import numpy as np
import cProfile

def main():
    tissue = Tissue(20, 20, 0.1)
    tissue.set_tracking()
    tissue.set_center_only(True)
    tissue.random_layout()
    tissue.set_plot_force_vector()
    tissue.simulate(f"Tissue annealing.", 1000, 100)
    tissue.set_flow([0, 1], 1)
    tissue.simulate("Tissue dynamics with cilia force.", 5000, 1000)
    tissue.write_to_file("./test.json")

#cProfile.run('main()', sort="tottime")
main()
