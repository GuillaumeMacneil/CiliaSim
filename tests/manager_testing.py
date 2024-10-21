from manager import Manager 

def main():
    manager = Manager()
    manager.read_from_file("saved_simulations/test29.json")
    manager.interactive_plot(0, "Interactive Plot")
    manager.energy_progression_plot(0, "Net Energy")
#    manager.duplicate_tissue(0)
#    manager.duplicate_tissue(0)
#    manager.duplicate_tissue(0)
#    manager.batch_load_iteration(2000)
#    manager.batch_simulate()

main()