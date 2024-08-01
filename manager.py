from tissue import Tissue
from plotting import *

from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import json
import multiprocessing as mp

# GLOBAL VARIABLE:
plot_type = 0

class LoadedTissue():
    def __init__(self, parameters: dict, cell_types: list, target_areas: list, cell_states: dict, force_states: dict, net_energy: list):
        # Keep the original __init__ parameters around for easier duplication
        self.init_parameters = parameters
        self.init_cell_types = cell_types
        self.init_target_areas = target_areas
        self.init_cell_states = cell_states
        self.init_force_states = force_states
        self.init_net_energy = net_energy

        self.tissue = Tissue(parameters["x"], parameters["y"], parameters["cilia_density"])
        self.tissue.target_areas = np.array(target_areas)
        self.cell_types = np.array(cell_types)
        self.cell_states = cell_states
        self.force_states = force_states
        self.net_energy = net_energy

        keys = self.cell_states.keys()
        keys = [int(key) for key in keys]
        self.min_it = min(keys)
        self.max_it = max(keys)
        self.current_it = self.min_it

        self.load_iteration(self.current_it)

    def load_iteration(self, iteration: int):
        if self.cell_states[str(iteration)]:
            self.tissue.cell_points = np.array(self.cell_states[str(iteration)])
            self.tissue.num_cells = len(self.tissue.cell_points)
            self.tissue.cell_types = self.cell_types[:self.tissue.num_cells]

            delaunay = Delaunay(self.tissue.cell_points)
            neighbour_vertices = delaunay.vertex_neighbor_vertices
            self.tissue.adjacency_matrix = np.zeros((self.tissue.num_cells,  self.tissue.num_cells))
            for i in range(self.tissue.num_cells):
                neighbours = neighbour_vertices[1][neighbour_vertices[0][i]:neighbour_vertices[0][i+1]]
                self.tissue.adjacency_matrix[i, neighbours] = 1

            # Boundary vertex conditions
            boundary_cells = np.where(self.tissue.cell_types == 1)[0]
            for cell in boundary_cells:
                cell_neighbours = np.where(self.tissue.adjacency_matrix[cell] == 1)
                boundary_neighbours = np.intersect1d(cell_neighbours, boundary_cells)
                differences = self.tissue.cell_points[boundary_neighbours] - self.tissue.cell_points[cell]
                distances = np.linalg.norm(differences, axis=1)
                sorted_boundary_neighbours = boundary_neighbours[np.argsort(distances)]
                furthest_neighbours = sorted_boundary_neighbours[2:]

                for neighbour in furthest_neighbours:
                    self.tissue.adjacency_matrix[cell, neighbour] = 0
                    self.tissue.adjacency_matrix[neighbour, cell] = 0

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
                self.tissue.set_uniform_cilia_forces([0, 0], 0)
            else:
                for key in self.force_states[closest].keys():
                    int_key = int(key)
                    if int_key == -1:
                        self.tissue.set_flow(self.force_states[closest][key], 1)
                    else:
                        self.tissue.add_cilia_force(int_key, self.force_states[closest][key])

            self.current_it = iteration
        else:
            raise ValueError(f"Iteration {iteration} does not lie in the range [{self.min_it}, {self.max_it}]")

    def set_flow(self,flow_direction: list, flow_magnitude: float):
        self.tissue.set_flow(flow_direction, flow_magnitude)

    def simulate(self, title: str, iterations: int = 5000, plotting: bool = True):
        self.tissue.simulate(title, iterations, plotting=plotting)

    def plot_tissue(self, title: str, duration: float, x_lim: int = 0, y_lim: int = 0, auto: bool = True):
        information = f"Iteration: {self.current_it}\nEx. Flow Force: {self.tissue.flow_force}"
        plot_tissue(self.tissue.cell_points, self.tissue.cell_types, title, duration, self.tissue.plot, x_lim=x_lim, y_lim=y_lim, information=information, auto=auto)

    def plot_springs(self, title: str, duration: float, x_lim: int = 0, y_lim: int = 0, auto: bool = True):
        information = f"Iteration: {self.current_it}\nEx. Flow Force: {self.tissue.flow_force}"
        plot_springs(self.tissue.cell_points, self.tissue.cell_types, self.tissue.adjacency_matrix, title, duration, self.tissue.plot, x_lim=x_lim, y_lim=y_lim, information=information, auto=auto)

    def plot_force_vectors_rel(self, title: str, duration: float, x_lim: int = 0, y_lim: int = 0, auto: bool = True):
        information = f"Iteration: {self.current_it}\nEx. Flow Force: {self.tissue.flow_force}"
        self.tissue.calculate_force_matrix()
        plot_force_vectors_rel(self.tissue.cell_points, self.tissue.cell_types, self.tissue.force_matrix, title, duration, self.tissue.plot, x_lim=x_lim, y_lim=y_lim, information=information, auto=auto)

    def plot_force_vectors_abs(self, title: str, duration: float, x_lim: int = 0, y_lim: int = 0, auto: bool = True):
        information = f"Iteration: {self.current_it}\nEx. Flow Force: {self.tissue.flow_force}"
        self.tissue.calculate_force_matrix()
        plot_force_vectors_abs(self.tissue.cell_points, self.tissue.cell_types, self.tissue.force_matrix, title, duration, self.tissue.plot, x_lim=x_lim, y_lim=y_lim, information=information, auto=auto)

    def plot_major_axes(self, title: str, duration: float, x_lim: int = 0, y_lim: int = 0, auto: bool = True):
        information = f"Iteration: {self.current_it}\nEx. Flow Force: {self.tissue.flow_force}"
        plot_major_axes(self.tissue.cell_points, self.tissue.cell_types, title, duration, self.tissue.plot, x_lim=x_lim, y_lim=y_lim, information=information, auto=auto)

    def plot_avg_major_axes(self, title: str, duration: float, x_lim: int = 0, y_lim: int = 0, auto: bool = True):
        information = f"Iteration: {self.current_it}\nEx. Flow Force: {self.tissue.flow_force}"
        plot_avg_major_axes(self.tissue.cell_points, self.tissue.cell_types, self.tissue.adjacency_matrix, title, duration, self.tissue.plot, x_lim=x_lim, y_lim=y_lim, information=information, auto=auto)

    def plot_area_deltas(self, title: str, duration: float, auto: bool = True):
        information = f"Iteration: {self.current_it}\nEx. Flow Force: {self.tissue.flow_force}"
        plot_area_delta(self.tissue.cell_points, self.tissue.cell_types, self.tissue.target_cell_area, title, duration, self.tissue.plot, information=information, auto=auto)
        
    def plot_neighbour_histogram(self, title: str, duration: float, auto: bool = True):
        information = f"Iteration: {self.current_it}\nEx. Flow Force: {self.tissue.flow_force}"
        plot_neighbour_histogram(self.tissue.adjacency_matrix, title, duration, self.tissue.plot, information=information, auto=auto)

    def plot_shape_factor_histogram(self, title: str, duration: float, auto: bool = True):
        information = f"Iteration: {self.current_it}\nEx. Flow Force: {self.tissue.flow_force}"
        shape_factors = self.tissue.calculate_shape_factors()
        plot_shape_factor_histogram(shape_factors, title, duration, self.tissue.plot, information=information, auto=auto)

    def plot_anisotropy_histogram(self, title: str, duration: float, auto: bool = True):
        information = f"Iteration: {self.current_it}\nEx. Flow Force: {self.tissue.flow_force}"
        plot_anisotropy_histogram(self.tissue.cell_points, self.tissue.cell_types, title, duration, self.tissue.plot, information=information, auto=auto)

    def plot_Q_divergence(self, title: str, duration: float, auto: bool = True):
        information = f"Iteration: {self.current_it}\nEx. Flow Force: {self.tissue.flow_force}"
        plot_Q_divergence(self.tissue.cell_points, self.tissue.cell_types, title, duration, self.tissue.plot, information=information, auto=auto)


    def interactive_tissue(self, title: str, start_iteration: int = 0, end_iteration: int = 0):
        def select_plot(plot_type: int):
            if plot_type == 0:
                self.plot_tissue(title, duration=0.1, auto=False)
            elif plot_type == 1:
                self.plot_springs(title, duration=0.1, auto=False)
            elif plot_type == 2:
                self.plot_force_vectors_rel(title, duration=0.1, auto=False)
            elif plot_type == 3:
                self.plot_force_vectors_abs(title, duration=0.1, auto=False)
            elif plot_type == 4:
                self.plot_major_axes(title, duration=0.1, auto=False)
            elif plot_type == 5:
                self.plot_avg_major_axes(title, duration=0.1, auto=False)
            elif plot_type == 6:
                self.plot_area_deltas(title, duration=0.1, auto=False)
            elif plot_type == 7:
                self.plot_neighbour_histogram(title, duration=0.1, auto=False)
            elif plot_type == 8:
                self.plot_shape_factor_histogram(title, duration=0.1, auto=False)
            elif plot_type == 9:
                self.plot_anisotropy_histogram(title, duration=0.1, auto=False)
            elif plot_type == 10:
                self.plot_Q_divergence(title, duration=0.1, auto=False)


        if start_iteration == 0:
            start_iteration = int(self.min_it)
        
        if end_iteration == 0:
            end_iteration = int(self.max_it)
        
        self.load_iteration(start_iteration)
        select_plot(plot_type)

        plt.subplots_adjust(left=0.1, bottom=0.25)

        slider_axis = self.tissue.plot.fig.add_axes([0.15, 0.15, 0.7, 0.03])

        iteration_slider = Slider(
            ax=slider_axis,
            label="Iteration",
            valmin=start_iteration,
            valmax=end_iteration,
            valinit=start_iteration,
        )

        basic_button_axis = self.tissue.plot.fig.add_axes([0.15, 0.1, 0.09, 0.03])
        basic_button = Button(basic_button_axis, "Basic")
        spring_button_axis = self.tissue.plot.fig.add_axes([0.25, 0.1, 0.09, 0.03])
        spring_button = Button(spring_button_axis, "Spring")
        force_rel_button_axis = self.tissue.plot.fig.add_axes([0.35, 0.1, 0.09, 0.03])
        force_rel_button = Button(force_rel_button_axis, "Rel. Force")
        force_abs_button_axis = self.tissue.plot.fig.add_axes([0.45, 0.1, 0.09, 0.03])
        force_abs_button = Button(force_abs_button_axis, "Abs. Force")
        major_axes_button_axis = self.tissue.plot.fig.add_axes([0.55, 0.1, 0.09, 0.03])
        major_axes_button = Button(major_axes_button_axis, "M. Axes")
        avg_major_axes_button_axis = self.tissue.plot.fig.add_axes([0.65, 0.1, 0.09, 0.03])
        avg_major_axes_button = Button(avg_major_axes_button_axis, "Avg. Axes")
        area_button_axis = self.tissue.plot.fig.add_axes([0.75, 0.1, 0.09, 0.03])
        area_button = Button(area_button_axis, "Area")
        neighbour_button_axis = self.tissue.plot.fig.add_axes([0.15, 0.05, 0.09, 0.03])
        neighbour_button = Button(neighbour_button_axis, "Conn.")
        shape_factor_button_axis = self.tissue.plot.fig.add_axes([0.25, 0.05, 0.09, 0.03])
        shape_factor_button = Button(shape_factor_button_axis, "S. Factor")
        anisotropy_button_axis = self.tissue.plot.fig.add_axes([0.35, 0.05, 0.09, 0.03])
        anisotropy_button = Button(anisotropy_button_axis, "Anisotropy")
        divergence_button_axis = self.tissue.plot.fig.add_axes([0.45, 0.05, 0.09, 0.03])
        divergence_button = Button(divergence_button_axis, "Q Div.")

        def button_plot(button_type: int):
            global plot_type
            plot_type = button_type
            select_plot(plot_type)
            self.tissue.plot.fig.canvas.draw_idle()

        basic_button.on_clicked(lambda _: button_plot(0))
        spring_button.on_clicked(lambda _: button_plot(1))
        force_rel_button.on_clicked(lambda _: button_plot(2))
        force_abs_button.on_clicked(lambda _: button_plot(3))
        major_axes_button.on_clicked(lambda _: button_plot(4))
        avg_major_axes_button.on_clicked(lambda _: button_plot(5))
        area_button.on_clicked(lambda _: button_plot(6))
        neighbour_button.on_clicked(lambda _: button_plot(7))
        shape_factor_button.on_clicked(lambda _: button_plot(8))
        anisotropy_button.on_clicked(lambda _: button_plot(9))
        divergence_button.on_clicked(lambda _: button_plot(10))

        def update_slider(_):
            global plot_type
            iteration = int(iteration_slider.val)
            self.load_iteration(iteration)
            select_plot(plot_type)
            self.tissue.plot.fig.canvas.draw_idle()

        def on_release(event):
            if event.inaxes == iteration_slider.ax:
                update_slider(event)

        def on_key(event):
            if event.key == 'right':
                new_val = min(iteration_slider.val + 10, iteration_slider.valmax)
                iteration_slider.set_val(new_val)
                update_slider(event)
            elif event.key == 'left':
                new_val = max(iteration_slider.val - 10, iteration_slider.valmin)
                iteration_slider.set_val(new_val)
                update_slider(event)
            elif event.key == 'shift+right':
                new_val = min(iteration_slider.val + 100, iteration_slider.valmax)
                iteration_slider.set_val(new_val)
                update_slider(event)
            elif event.key == 'shift+left':
                new_val = max(iteration_slider.val - 100, iteration_slider.valmin)
                iteration_slider.set_val(new_val)
                update_slider(event)
            elif event.key == 'q':
                return None

        self.tissue.plot.fig.canvas.mpl_connect('button_release_event', on_release)
        self.tissue.plot.fig.canvas.mpl_connect('key_press_event', on_key)

        plt.show()

    def animate_tissue(self, title: str, plot_type: int, start_iteration: int = 0, end_iteration: int = 0, step: int = 50):
        plt.ion()

        if start_iteration == 0:
            start_iteration = int(self.min_it)
        
        if end_iteration == 0:
            end_iteration = int(self.max_it)

        for i in range(0, end_iteration - start_iteration, step):
            self.load_iteration(start_iteration + i)
            if plot_type == 0:
                self.plot_tissue(title, duration=0.1)
            elif plot_type == 1:
                self.plot_springs(title, duration=0.1)
            elif plot_type == 2:
                self.plot_force_vectors_rel(title, duration=0.1)
            elif plot_type == 3:
                self.plot_force_vectors_abs(title, duration=0.1)
            elif plot_type == 4:
                self.plot_major_axes(title, duration=0.1)
            elif plot_type == 5:
                self.plot_avg_major_axes(title, duration=0.1)
            elif plot_type == 6:
                self.plot_area_deltas(title, duration=0.1)
            elif plot_type == 7:
                self.plot_neighbour_histogram(title, duration=0.1)
            elif plot_type == 8:
                self.plot_shape_factor_histogram(title, duration=0.1)
            elif plot_type == 9:
                self.plot_anisotropy_histogram(title, duration=0.1)
            elif plot_type == 10:
                self.plot_Q_divergence(title, duration=0.1, auto=False)


    def plot_energy_progression(self, title: str):
        fig, ax = plt.subplots()
        ax.plot(np.arange(0, len(self.net_energy)), self.net_energy)

        fig.set_figheight(8)
        fig.set_figwidth(8)
        
        plt.title(title)
        plt.show()

class Manager():
    def __init__(self):
        self.tissues = []

    def read_from_file(self, path: str):
        with open(path, "r") as input_file:
            json_data = json.load(input_file)

            parameters = json_data["parameters"]
            cell_types = json_data["cell_types"]
            target_areas = json_data["target_areas"]
            cell_states = json_data["cell_states"]
            force_states = json_data["force_states"]
            net_energy = json_data["net_energy"]

            self.tissues.append(LoadedTissue(parameters, cell_types, target_areas, cell_states, force_states, net_energy))

    def interactive_plot(self, index: int, title: str, start_iteration: int = 0, end_iteration: int = 0):
        self.tissues[index].interactive_tissue(title, start_iteration, end_iteration)

    def animate_plot(self, index: int, title: str, plot_type: int, start_iteration: int = 0, end_iteration: int = 0, step: int = 50):
        self.tissues[index].animate_tissue(title, plot_type, start_iteration, end_iteration, step)

    def energy_progression_plot(self, index: int, title: str):
        self.tissues[index].plot_energy_progression(title)

    def duplicate_tissue(self, index: int):
        from_tissue = self.tissues[index]
        self.tissues.append(
            LoadedTissue(
                from_tissue.init_parameters,
                from_tissue.init_cell_types,
                from_tissue.init_target_areas,
                from_tissue.init_cell_states,
                from_tissue.init_force_states,
                from_tissue.init_net_energy,
            )
        )

    def batch_load_iteration(self, iteration: int):
        for tissue in self.tissues:
            if not (tissue.min_it <= iteration <= tissue.max_it):
                raise ValueError(f"Batch load iteration must be valid for all tissues ({tissue.min_it} <= x <= {tissue.max_it})")
        
        for tissue in self.tissues:
            tissue.load_iteration(iteration)

    
    def batch_simulate(self):
        with mp.Pool(processes=mp.cpu_count()) as pool:
            pool.map(simulate_tissue, self.tissues)

def simulate_tissue(tissue):
    tissue.simulate("", 1000, plotting=False)


