from CiliaSim.tissue import Tissue
from time import time


def main():
    t1 = time()
    tissue = Tissue(15, 15, 0.06)
    tissue.set_tracking()
    # tissue.set_plotting()
    tissue.set_center_only(True)
    tissue.hexagonal_grid_layout()
    tissue.simulate(
        title="Tissue annealing - No cilia force.",
        dt=0.01,
        damping=0.95,
        iterations=1000,
        plot_frequency=100,
        plotting=False,
    )
    tissue.set_uniform_cilia_forces([0, 1], 0.4)
    tissue.simulate(
        title=f"Tissue under random cilia force of mag. {0.5}.",
        dt=0.01,
        damping=0.95,
        iterations=5000,
        plot_frequency=100,
        plotting=False,
    )

    tissue.write_to_file("./saved_simulations/test30.json")
    t2 = time()
    print(f"Time taken: {t2 - t1} seconds.")


# cProfile.run('main()', sort="tottime")
main()
