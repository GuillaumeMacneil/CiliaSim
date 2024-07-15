from tissue import Tissue
import numpy as np
import cProfile

def main():
    tissue = Tissue(15, 15, 0.1)
    tissue.set_tracking()
    tissue.set_center_only(True)
    tissue.random_layout()
    tissue.simulate(f"Tissue annealing - 3000 iterations.", 2000, annealing=True)
    tissue.set_flow([0, 1], 0.5)
    tissue.set_plot_spring()
    tissue.simulate("Tissue dynamics with cilia force.", 1000)
    tissue.write_to_file("./test.json")

#cProfile.run('main()', sort="tottime")
main()
