from ciliasim import Tissue


def main():
    tissue = Tissue(15, 15, 0.06)
    tissue.set_tracking()
    # tissue.set_plotting()
    tissue.set_center_only(True)
    tissue.hexagonal_grid_layout()
    # tissue.set_plot_major_axes()
    # tissue.set_plot_spring()
    tissue.simulate(f"Tissue annealing - No cilia force.", 1000, 100, plotting=False)
    tissue.set_uniform_cilia_forces([0, 1], 0.5)
    # tissue.set_random_cilia_forces(0.5)
    tissue.simulate(
        f"Tissue under random cilia force of mag. {0.5}.", 5000, 100, plotting=False
    )

    tissue.write_to_file("./saved_simulations/test30.json")


# cProfile.run('main()', sort="tottime")
main()
