from tissue import Tissue
import numpy as np
import cProfile

def main():
    tissue = Tissue(10, 10, 0.1)
    #tissue.set_tracking()
    tissue.set_center_only(True)
    tissue.hexagonal_grid_layout()
    tissue.set_plot_area_deltas()
    tissue.simulate(f"Tissue annealing.", 1000, 100)
    tissue.set_flow([1, 0], 2)
    tissue.simulate("Tissue dynamics with cilia force.", 10000, 10)
    #tissue.write_to_file("./test.json")

#cProfile.run('main()', sort="tottime")
main()
