from functions import *

import matplotlib.pyplot as plt
import matplotlib.colors as mcolours
import matplotlib.cm as cm
import numpy as np
from scipy.spatial import Voronoi

class TissuePlot():
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.fig.set_figwidth(8)
        self.fig.set_figheight(8)
        self.colourbar = None
        self.information_box = self.fig.text(
            0.05,
            0.95,
            "",
            transform = self.ax.transAxes,
            fontsize = 10,
            verticalalignment= "top",
            bbox = dict(boxstyle="round", facecolor="white", alpha=0.5)
        )

def plot_tissue(points: np.ndarray, cell_types: np.ndarray, title: str, duration: float, plot: TissuePlot, x_lim: int = 0, y_lim: int = 0, information: str = "", auto: bool = True):
    plot.ax.clear()

    basic_indices = np.where(cell_types == 0)[0]
    boundary_indices = np.where(cell_types == 1)[0]
    boundary_points = points[boundary_indices]
    multiciliated_indices = np.where(cell_types == 2)[0]

    voronoi = Voronoi(points)

    for basic_index in basic_indices:
        region = voronoi.regions[voronoi.point_region[basic_index]]
        if -1 not in region:
            polygon = voronoi.vertices[region]
            plot.ax.fill(*zip(*polygon), alpha=0.6, color="lightgrey", edgecolor="black")

    for multiciliated_index in multiciliated_indices:
        region = voronoi.regions[voronoi.point_region[multiciliated_index]]
        if -1 not in region:
            polygon = voronoi.vertices[region]
            plot.ax.fill(*zip(*polygon), alpha=0.6, color="orange", edgecolor="black")

    plot.ax.scatter(boundary_points[:, 0], boundary_points[:, 1], s=20, color="green")   
    
    plot.information_box.set_text(information)
    if plot.colourbar:
        if plot.colourbar.ax.get_visible():
            plot.colourbar.ax.set_visible(False)
            ax_pos = plot.ax.get_position()
            plot.ax.set_position([ax_pos.x0, ax_pos.y0, ax_pos.width + 0.1, ax_pos.height])

    if x_lim and y_lim:
        plt.xlim([0, x_lim])
        plt.ylim([0, y_lim])

    plot.ax.set_title(title)

    if auto:
        plt.show()
        plt.pause(duration)

def plot_springs(points: np.ndarray, cell_types: np.ndarray, adjacency_matrix: np.ndarray, title: str, duration: float, plot: TissuePlot, x_lim: int = 0, y_lim: int = 0, information: str = "", auto: bool = True):
    plot.ax.clear()

    basic_indices = np.where(cell_types == 0)[0]
    boundary_indices = np.where(cell_types == 1)[0]
    boundary_points = points[boundary_indices]
    multiciliated_indices = np.where(cell_types == 2)[0]

    voronoi = Voronoi(points)

    for basic_index in basic_indices:
        region = voronoi.regions[voronoi.point_region[basic_index]]
        if -1 not in region:
            polygon = voronoi.vertices[region]
            plot.ax.fill(*zip(*polygon), alpha=0.6, color="lightgrey", edgecolor="black")

    for multiciliated_index in multiciliated_indices:
        region = voronoi.regions[voronoi.point_region[multiciliated_index]]
        if -1 not in region:
            polygon = voronoi.vertices[region]
            plot.ax.fill(*zip(*polygon), alpha=0.6, color="orange", edgecolor="black")

    plot.ax.scatter(boundary_points[:, 0], boundary_points[:, 1], s=20, color="green")   

    distance_matrix = np.zeros((len(points), len(points)))
    for i in range(len(points)):
        neighbours = np.where(adjacency_matrix[i] == 1)
        neighbour_points = points[neighbours]
        distances = np.linalg.norm(neighbour_points - points[i], axis=1) - 1
        distance_matrix[i, neighbours] = distances

    colourmap = cm.plasma
    norm = mcolours.Normalize(vmin=np.min(distance_matrix), vmax=np.max(distance_matrix))
    colour_matrix = colourmap(norm(distance_matrix))

    for i in range(len(points)):
        neighbours = np.where(adjacency_matrix[i] == 1)
        neighbour_points = points[neighbours]
        for j in range(len(neighbours[0])):
            plot.ax.plot([points[i, 0], neighbour_points[j, 0]], [points[i, 1], neighbour_points[j, 1]], color=colour_matrix[i][neighbours][j])

    plot.information_box.set_text(information)

    if not plot.colourbar:
        plot.colourbar = plot.fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colourmap), ax=plot.ax, orientation="vertical")
    else:
        plot.colourbar.update_normal(cm.ScalarMappable(norm=norm, cmap=colourmap))
        if not plot.colourbar.ax.get_visible():
            ax_pos = plot.ax.get_position()
            plot.ax.set_position([ax_pos.x0, ax_pos.y0, ax_pos.width - 0.1, ax_pos.height])
            plot.colourbar.ax.set_visible(True)

    if x_lim and y_lim:
        plt.xlim([0, x_lim])
        plt.ylim([0, y_lim])

    plot.ax.set_title(title)

    if auto:
        plt.show()
        plt.pause(duration)


def plot_force_vectors(points: np.ndarray, cell_types: np.ndarray, force_matrix: np.ndarray, title: str, duration: float, plot: TissuePlot, x_lim: int = 0, y_lim: int = 0, information: str = "", auto: bool = True):
    plot.ax.clear()

    basic_indices = np.where(cell_types == 0)[0]
    boundary_indices = np.where(cell_types == 1)[0]
    boundary_points = points[boundary_indices]
    multiciliated_indices = np.where(cell_types == 2)[0]

    voronoi = Voronoi(points)

    for basic_index in basic_indices:
        region = voronoi.regions[voronoi.point_region[basic_index]]
        if -1 not in region:
            polygon = voronoi.vertices[region]
            plot.ax.fill(*zip(*polygon), alpha=0.6, color="lightgrey", edgecolor="black")

    for multiciliated_index in multiciliated_indices:
        region = voronoi.regions[voronoi.point_region[multiciliated_index]]
        if -1 not in region:
            polygon = voronoi.vertices[region]
            plot.ax.fill(*zip(*polygon), alpha=0.6, color="orange", edgecolor="black")

    plot.ax.scatter(boundary_points[:, 0], boundary_points[:, 1], s=20, color="green")   

    force_vector = np.array([np.sum(force_matrix[:, point_index], axis=0) for point_index in range(len(points))])
    plot.ax.quiver(points[:, 0], points[:, 1], force_vector[:, 0], force_vector[:, 1])

    plot.information_box.set_text(information)
    if plot.colourbar:
        if plot.colourbar.ax.get_visible():
            plot.colourbar.ax.set_visible(False)
            ax_pos = plot.ax.get_position()
            plot.ax.set_position([ax_pos.x0, ax_pos.y0, ax_pos.width + 0.1, ax_pos.height])

    if x_lim and y_lim:
        plt.xlim([0, x_lim])
        plt.ylim([0, y_lim])

    plot.ax.set_title(title)

    if auto:
        plt.show()
        plt.pause(duration)
 
def plot_major_axes(points: np.ndarray, cell_types: np.ndarray, title: str, duration: float, plot: TissuePlot, x_lim: int = 0, y_lim: int = 0, information: str = "", auto: bool = True):
    plot.ax.clear()

    basic_indices = np.where(cell_types == 0)[0]
    boundary_indices = np.where(cell_types == 1)[0]
    boundary_points = points[boundary_indices]
    multiciliated_indices = np.where(cell_types == 2)[0]

    voronoi = Voronoi(points)

    for basic_index in basic_indices:
        region = voronoi.regions[voronoi.point_region[basic_index]] 
        if -1 not in region:
            polygon = voronoi.vertices[region]
            plot.ax.fill(*zip(*polygon), alpha=0.6, color="lightgrey", edgecolor="black")

            center = points[basic_index]
            centered_points = polygon - center 
            covariance_matrix = np.cov(centered_points, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
            major_axis_index = np.argmax(eigenvalues)
            major_axis = eigenvectors[:, major_axis_index]
            
            projections = np.dot(centered_points, major_axis)
            length = projections.max() - projections.min()

            positive_end = center + (length / 2) * major_axis
            negative_end = center - (length / 2) * major_axis

            plot.ax.plot([negative_end[0], positive_end[0]], [negative_end[1], positive_end[1]], linewidth=2, color="red")

    for multiciliated_index in multiciliated_indices:
        region = voronoi.regions[voronoi.point_region[multiciliated_index]]
        if -1 not in region:
            polygon = voronoi.vertices[region]
            plot.ax.fill(*zip(*polygon), alpha=0.6, color="orange", edgecolor="black")

            center = points[multiciliated_index]
            centered_points = polygon - center 
            covariance_matrix = np.cov(centered_points, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
            major_axis_index = np.argmax(eigenvalues)
            major_axis = eigenvectors[:, major_axis_index]
            
            projections = np.dot(centered_points, major_axis)
            length = projections.max() - projections.min()

            positive_end = center + (length / 2) * major_axis
            negative_end = center - (length / 2) * major_axis

            plot.ax.plot([negative_end[0], positive_end[0]], [negative_end[1], positive_end[1]], linewidth=2, color="red")

    plot.ax.scatter(boundary_points[:, 0], boundary_points[:, 1], s=20, color="green")   

    plot.information_box.set_text(information)
    if plot.colourbar:
        if plot.colourbar.ax.get_visible():
            plot.colourbar.ax.set_visible(False)
            ax_pos = plot.ax.get_position()
            plot.ax.set_position([ax_pos.x0, ax_pos.y0, ax_pos.width + 0.1, ax_pos.height])

    if x_lim and y_lim:
        plt.xlim([0, x_lim])
        plt.ylim([0, y_lim])

    plot.ax.set_title(title)

    if auto:
        plt.show()
        plt.pause(duration)

def plot_avg_major_axes(points: np.ndarray, cell_types: np.ndarray, adjacency_matrix: np.ndarray, title: str, duration: float, plot: TissuePlot, x_lim: int = 0, y_lim: int = 0, information: str = "", auto: bool = True):
    plot.ax.clear()

    basic_indices = np.where(cell_types == 0)[0]
    boundary_indices = np.where(cell_types == 1)[0]
    boundary_points = points[boundary_indices]
    multiciliated_indices = np.where(cell_types == 2)[0]

    voronoi = Voronoi(points)

    for basic_index in basic_indices:
        region = voronoi.regions[voronoi.point_region[basic_index]]
        if -1 not in region:
            polygon = voronoi.vertices[region]
            plot.ax.fill(*zip(*polygon), alpha=0.6, color="lightgrey", edgecolor="black")

            major_axes = []
            for neighbour in np.where(adjacency_matrix[basic_index] == 1)[0]:
                center = points[neighbour] 
                centered_points = polygon - center 
                covariance_matrix = np.cov(centered_points, rowvar=False)
                eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
                major_axis_index = np.argmax(eigenvalues)
                major_axis = eigenvectors[:, major_axis_index]
                major_axes.append(major_axis)    

            avg_major_axis = np.mean(major_axes, axis=0)
            center = points[basic_index] 
            centered_points = polygon - center 

            projections = np.dot(centered_points, avg_major_axis)
            length = projections.max() - projections.min()

            positive_end = center + (length / 2) * avg_major_axis
            negative_end = center - (length / 2) * avg_major_axis

            plot.ax.plot([negative_end[0], positive_end[0]], [negative_end[1], positive_end[1]], linewidth=2, color="red")

    for multiciliated_index in multiciliated_indices:
        region = voronoi.regions[voronoi.point_region[multiciliated_index]]
        if -1 not in region:
            polygon = voronoi.vertices[region]
            plot.ax.fill(*zip(*polygon), alpha=0.6, color="orange", edgecolor="black")

            major_axes = []
            for neighbour in np.where(adjacency_matrix[multiciliated_index] == 1)[0]:
                center = points[neighbour] 
                centered_points = polygon - center 
                covariance_matrix = np.cov(centered_points, rowvar=False)
                eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
                major_axis_index = np.argmax(eigenvalues)
                major_axis = eigenvectors[:, major_axis_index]
                major_axes.append(major_axis)    

            avg_major_axis = np.mean(major_axes, axis=0)
            center = points[multiciliated_index] 
            centered_points = polygon - center 

            projections = np.dot(centered_points, avg_major_axis)
            length = projections.max() - projections.min()

            positive_end = center + (length / 2) * avg_major_axis
            negative_end = center - (length / 2) * avg_major_axis

            plot.ax.plot([negative_end[0], positive_end[0]], [negative_end[1], positive_end[1]], linewidth=2, color="red")

    plot.ax.scatter(boundary_points[:, 0], boundary_points[:, 1], s=20, color="green")   

    plot.information_box.set_text(information)
    if plot.colourbar:
        if plot.colourbar.ax.get_visible():
            plot.colourbar.ax.set_visible(False)
            ax_pos = plot.ax.get_position()
            plot.ax.set_position([ax_pos.x0, ax_pos.y0, ax_pos.width + 0.1, ax_pos.height])

    if x_lim and y_lim:
        plt.xlim([0, x_lim])
        plt.ylim([0, y_lim])

    plot.ax.set_title(title)
    
    if auto:
        plt.show()
        plt.pause(duration)

def plot_area_delta(points: np.ndarray, cell_types: np.ndarray, target_area: float, title: str, duration: float, plot: TissuePlot, information: str = "", auto: bool = True):
    plot.ax.clear()

    voronoi = Voronoi(points)

    area_diffs = []
    non_border_indices = np.where(cell_types != 1)[0]
    for i in non_border_indices:
        area = polygon_area(voronoi.vertices[voronoi.regions[voronoi.point_region[i]]]) 
        area_diffs.append(area - target_area)

    plot.ax.hist(area_diffs)

    plot.information_box.set_text(information)
    if plot.colourbar:
        if plot.colourbar.ax.get_visible():
            plot.colourbar.ax.set_visible(False)
            ax_pos = plot.ax.get_position()
            plot.ax.set_position([ax_pos.x0, ax_pos.y0, ax_pos.width + 0.1, ax_pos.height])

    plot.ax.set_title(title)

    if auto:
        plt.show()
        plt.pause(duration)
    
def plot_neighbour_histogram(adjacency_matrix: np.ndarray, title: str, duration: float, plot: TissuePlot, information: str = "", auto: bool = True):
    plot.ax.clear()

    connectivity = np.sum(adjacency_matrix, axis=0)
    disc_size = np.diff(np.unique(connectivity)).min()
    left = connectivity.min() - float(disc_size) / 2
    right = connectivity.max() + float(disc_size) / 2

    plot.ax.hist(connectivity, np.arange(left, right + disc_size, disc_size))
    
    plot.information_box.set_text(information)
    if plot.colourbar:
        if plot.colourbar.ax.get_visible():
            plot.colourbar.ax.set_visible(False)
            ax_pos = plot.ax.get_position()
            plot.ax.set_position([ax_pos.x0, ax_pos.y0, ax_pos.width + 0.1, ax_pos.height])

    plot.ax.set_title(title)

    if auto:
        plt.show()
        plt.pause(duration)
    
def plot_shape_factor_histogram(shape_factors: np.ndarray, title: str, duration: float, plot: TissuePlot, information: str = "", auto: bool = True):
    plot.ax.clear()

    plot.ax.hist(shape_factors, bins=20)
    max_sf = np.max(shape_factors)
    min_sf = np.min(shape_factors)

    shape_dict = {3.722: ["red", "Hexagon"],
                  3.812: ["green", "Pentagon"],
                  4.0: ["blue", "Square"],
                  4.559: ["orange", "Triangle"]
    }
    for key in shape_dict.keys():
        if min_sf <= key <= max_sf:
            plot.ax.axvline(x=key, color=shape_dict[key][0], linestyle="--", linewidth=1, label=shape_dict[key][1])

    plot.ax.legend()

    plot.information_box.set_text(information)
    if plot.colourbar:
        if plot.colourbar.ax.get_visible():
            plot.colourbar.ax.set_visible(False)
            ax_pos = plot.ax.get_position()
            plot.ax.set_position([ax_pos.x0, ax_pos.y0, ax_pos.width + 0.1, ax_pos.height])

    plot.ax.set_title(title)
    
    if auto:
        plt.show()
        plt.pause(duration)
    
