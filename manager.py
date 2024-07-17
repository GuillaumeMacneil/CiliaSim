from tissue import Tissue
from cell import BasicCell, MulticiliatedCell, BorderCell
from plotting import *

from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import json

class LoadedTissue():
    def __init__(self, parameters: dict, cell_inits: list, cell_states: dict, force_states: dict):
        self.tissue = Tissue(parameters["x"], parameters["y"], parameters["cilia_density"])
        self.cell_inits = cell_inits
        self.cell_states = cell_states
        self.force_states = force_states

        keys = self.cell_states.keys()
        keys = [int(key) for key in keys]
        self.min_it = min(keys)
        self.max_it = max(keys)
        self.current_it = self.min_it

        self.load_iteration(self.current_it)
        plt.ion()

    def load_iteration(self, iteration: int):
        # FIXME: Very inefficient to reset the cell list every time
        self.tissue.cells = []
        if self.cell_states[str(iteration)]:
            cell_positions = self.cell_states[str(iteration)]
            delaunay = Delaunay(cell_positions)
            neighbour_vertices = delaunay.vertex_neighbor_vertices
            for i in range(len(cell_positions)):
                neighbours = neighbour_vertices[1][neighbour_vertices[0][i]:neighbour_vertices[0][i+1]]
                if self.cell_inits[i][0] == 0:
                    self.tissue.cells.append(BasicCell(i, cell_positions[i][0], cell_positions[i][1], neighbours))
                    self.tissue.cells[-1].set_area(self.cell_inits[i][1])
                elif self.cell_inits[i][0] == 1:
                    self.tissue.cells.append(BorderCell(i, cell_positions[i][0], cell_positions[i][1], neighbours))
                else:
                    self.tissue.cells.append(MulticiliatedCell(i, cell_positions[i][0], cell_positions[i][1], neighbours))
                    self.tissue.cells[-1].set_area(self.cell_inits[i][1])

            # FIXME: This will need to be improved when the force definitions become more complicated
            closest = None
            for key in self.force_states.keys():
                if int(key) == int(iteration):
                    closest = key
                    break
                elif int(key) < int(iteration):
                    closest = key
                else:
                    break

            if closest == None:
                self.tissue.set_flow([0, 0], 0)
            else:
                self.tissue.set_flow(self.force_states[closest][1], self.force_states[closest][2])

            self.current_it = iteration
        else:
            raise ValueError(f"Iteration {iteration} does not lie in the range [{self.min_it}, {self.max_it}]")

    def set_flow(self,flow_direction: list, flow_magnitude: float):
        self.tissue.set_flow(flow_direction, flow_magnitude)

    def simulate(self, title: str, iterations: int = 5000):
        self.tissue.simulate(title, iterations)

    def plot_tissue(self, title: str, duration: float):
        information = f"Iteration: {self.current_it}\nCilia force magnitude: {self.tissue.flow_magnitude}\nCilia force direction: {self.tissue.flow_direction}"
        self.tissue.plot = plot_tissue(self.tissue.cells, title, duration, self.tissue.plot, information=information)

    def plot_springs(self, title: str, duration: float):
        information = f"Iteration: {self.current_it}\nCilia force magnitude: {self.tissue.flow_magnitude}\nCilia force direction: {self.tissue.flow_direction}"
        self.tissue.plot = plot_springs(self.tissue.cells, title, duration, self.tissue.plot, information=information)

    def plot_force_vectors(self, title: str, duration: float):
        information = f"Iteration: {self.current_it}\nCilia force magnitude: {self.tissue.flow_magnitude}\nCilia force direction: {self.tissue.flow_direction}"
        force_matrix = self.tissue.calculate_force_matrix()
        self.tissue.plot = plot_force_vectors(self.tissue.cells, force_matrix, title, duration, self.tissue.plot, information=information)

    def plot_major_axes(self, title: str, duration: float):
        information = f"Iteration: {self.current_it}\nCilia force magnitude: {self.tissue.flow_magnitude}\nCilia force direction: {self.tissue.flow_direction}"
        self.tissue.plot = plot_major_axes(self.tissue.cells, title, duration, self.tissue.plot, information=information)

    def plot_avg_major_axes(self, title: str, duration: float):
        information = f"Iteration: {self.current_it}\nCilia force magnitude: {self.tissue.flow_magnitude}\nCilia force direction: {self.tissue.flow_direction}"
        self.tissue.plot = plot_avg_major_axes(self.tissue.cells, title, duration, self.tissue.plot, information=information)

class Manager():
    def __init__(self):
        self.tissues = []

    def read_from_file(self, path: str):
        with open(path, "r") as input_file:
            json_data = json.load(input_file)

            parameters = json_data["parameters"]
            cell_inits = json_data["cell_inits"]
            cell_states = json_data["cell_states"]
            force_states = json_data["force_states"]

            self.tissues.append(LoadedTissue(parameters, cell_inits, cell_states, force_states))
            
    def animate_tissue(self, title: str, index: int, plot_type: int, start_iteration: int = 0, end_iteration: int = 0, step=100):
        tissue = self.tissues[index]

        if start_iteration == 0:
            start_iteration = int(tissue.min_it)
        
        if end_iteration == 0:
            end_iteration = int(tissue.max_it)
        
        for i in range(0, end_iteration - start_iteration, step):
            tissue.load_iteration(start_iteration + i)
            if plot_type == 0:
                tissue.plot_tissue(title, duration=0.1)
            elif plot_type == 1:
                tissue.plot_springs(title, duration=0.1)
            elif plot_type == 2:
                tissue.plot_force_vectors(title, duration=0.1)
            elif plot_type == 3:
                tissue.plot_major_axes(title, duration=0.1)
            elif plot_type == 4:
                tissue.plot_avg_major_axes(title, duration=0.1)


    
# TESTING
manager = Manager()
manager.read_from_file("./test.json")
manager.animate_tissue("Animating Tissue Progression", index=0, plot_type=2, step=10)

