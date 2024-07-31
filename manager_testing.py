from manager import Manager 

def main():
    manager = Manager()
    manager.read_from_file("saved_simulations/test16.json")
    manager.interactive_plot(0, "Interactive Plot")
#    manager.duplicate_tissue(0)
#    manager.duplicate_tissue(0)
#    manager.duplicate_tissue(0)
#    manager.batch_load_iteration(2000)
#    manager.batch_simulate()

main()
