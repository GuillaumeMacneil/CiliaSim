from cell import BasicCell, MulticiliatedCell, BorderCell

import matplotlib.pyplot as plt
import matplotlib.colors as mcolours
import matplotlib.cm as cm
import numpy as np
from scipy.spatial import Voronoi

def plot_tissue(cells: list, title: str, duration: float, plot: tuple = (None, None), x_lim: int = 0, y_lim: int = 0, information: str = ""):
    points = []
    basic_points = []
    basic_indices = []
    multiciliated_points = []
    multiciliated_indices = []
    boundary_points = []
    boundary_indices = []

    for i in range(len(cells)):
        points.append([cells[i].x, cells[i].y])
        if isinstance(cells[i], BasicCell):
            basic_points.append([cells[i].x, cells[i].y])
            basic_indices.append(i)
        elif isinstance(cells[i], BorderCell):
            boundary_points.append([cells[i].x, cells[i].y])
            boundary_indices.append(i)
        else:
            multiciliated_points.append([cells[i].x, cells[i].y])
            multiciliated_indices.append(i)

    voronoi = Voronoi(points)

    if not plot[0] and not plot[1]:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot[0], plot[1]

    for basic_index in basic_indices:
        region = voronoi.regions[voronoi.point_region[basic_index]]
        if -1 not in region:
            polygon = voronoi.vertices[region]
            ax.fill(*zip(*polygon), alpha=0.6, color="lightgrey")
            ax.fill(*zip(*polygon), alpha=0.6, edgecolor="black", fill=False)

    for multiciliated_index in multiciliated_indices:
        region = voronoi.regions[voronoi.point_region[multiciliated_index]]
        if -1 not in region:
            polygon = voronoi.vertices[region]
            ax.fill(*zip(*polygon), alpha=0.6, color="orange")
            ax.fill(*zip(*polygon), alpha=0.6, edgecolor="black", fill=False)

    if boundary_points:
        boundary_array = np.array(boundary_points)
        ax.scatter(boundary_array[:, 0], boundary_array[:, 1], s=20, color="green")   

    fig.set_figwidth(8)
    fig.set_figheight(8)
    
    if information:
        information_box = dict(boxstyle="round", facecolor="white")
        fig.text(0.05, 0.95, information, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=information_box)

    if x_lim and y_lim:
        plt.xlim([0, x_lim])
        plt.ylim([0, y_lim])

    plt.title(title)
    plt.show()
    plt.pause(duration)

    ax.clear()
    
    return (fig, ax, None)

def plot_springs(cells: list, force_matrix: np.ndarray, title: str, duration: float, plot: tuple = (None, None), x_lim: int = 0, y_lim: int = 0, information: str = ""):
    points = []
    basic_points = []
    basic_indices = []
    multiciliated_points = []
    multiciliated_indices = []
    boundary_points = []
    boundary_indices = []

    for i in range(len(cells)):
        points.append([cells[i].x, cells[i].y])
        if isinstance(cells[i], BasicCell):
            basic_points.append([cells[i].x, cells[i].y])
            basic_indices.append(i)
        elif isinstance(cells[i], BorderCell):
            boundary_points.append([cells[i].x, cells[i].y])
            boundary_indices.append(i)
        else:
            multiciliated_points.append([cells[i].x, cells[i].y])
            multiciliated_indices.append(i)

    voronoi = Voronoi(points)

    if not plot[0] and not plot[1]:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot[0], plot[1]

    if not plot[2]:
        colourbar = None
    else:
        colourbar = plot[2]
        
    magnitude_matrix = np.linalg.norm(force_matrix, axis=2)
    colourmap = cm.plasma
    norm = mcolours.Normalize(vmin=np.min(magnitude_matrix), vmax=np.max(magnitude_matrix))
    colour_matrix = colourmap(norm(magnitude_matrix))

    for basic_index in basic_indices:
        region = voronoi.regions[voronoi.point_region[basic_index]]
        if -1 not in region:
            polygon = voronoi.vertices[region]
            ax.fill(*zip(*polygon), alpha=0.6, color="lightgrey")
            ax.fill(*zip(*polygon), alpha=0.6, edgecolor="black", fill=False)

        for neighbour in cells[basic_index].neighbours:
            ax.plot([cells[basic_index].x, cells[neighbour].x], [cells[basic_index].y, cells[neighbour].y], linewidth=2, color=colour_matrix[basic_index, neighbour])

    for multiciliated_index in multiciliated_indices:
        region = voronoi.regions[voronoi.point_region[multiciliated_index]]
        if -1 not in region:
            polygon = voronoi.vertices[region]
            ax.fill(*zip(*polygon), alpha=0.6, color="orange")
            ax.fill(*zip(*polygon), alpha=0.6, edgecolor="black", fill=False)

        for neighbour in cells[multiciliated_index].neighbours:
            ax.plot([cells[multiciliated_index].x, cells[neighbour].x], [cells[multiciliated_index].y, cells[neighbour].y], linewidth=2, color=colour_matrix[multiciliated_index, neighbour])

    for boundary_index in boundary_indices:
        for neighbour in cells[boundary_index].neighbours:
            ax.plot([cells[boundary_index].x, cells[neighbour].x], [cells[boundary_index].y, cells[neighbour].y], linewidth=2, color=colour_matrix[boundary_index, neighbour])

    if boundary_points:
        boundary_array = np.array(boundary_points)
        ax.scatter(boundary_array[:, 0], boundary_array[:, 1], s=20, color="green")   

    fig.set_figwidth(10)
    fig.set_figheight(8)
    
    if information:
        information_box = dict(boxstyle="round", facecolor="white")
        fig.text(0.05, 0.95, information, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=information_box)

    if not colourbar:
        colourbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colourmap), ax=ax, orientation="vertical")
    else:
        colourbar.update_normal(cm.ScalarMappable(norm=norm, cmap=colourmap))

    
    if x_lim and y_lim:
        plt.xlim([0, x_lim])
        plt.ylim([0, y_lim])

    plt.title(title)
    plt.show()
    plt.pause(duration)

    ax.clear()
    
    return (fig, ax, colourbar)

def plot_force_vectors(cells: list, force_matrix: np.ndarray, title: str, duration: float, plot: tuple = (None, None), x_lim: int = 0, y_lim: int = 0, information: str = ""):
    points = []
    basic_points = []
    basic_indices = []
    multiciliated_points = []
    multiciliated_indices = []
    boundary_points = []
    boundary_indices = []

    for i in range(len(cells)):
        points.append([cells[i].x, cells[i].y])
        if isinstance(cells[i], BasicCell):
            basic_points.append([cells[i].x, cells[i].y])
            basic_indices.append(i)
        elif isinstance(cells[i], BorderCell):
            boundary_points.append([cells[i].x, cells[i].y])
            boundary_indices.append(i)
        else:
            multiciliated_points.append([cells[i].x, cells[i].y])
            multiciliated_indices.append(i)

    voronoi = Voronoi(points)

    if not plot[0] and not plot[1]:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot[0], plot[1]
        
    for basic_index in basic_indices:
        region = voronoi.regions[voronoi.point_region[basic_index]]
        if -1 not in region:
            polygon = voronoi.vertices[region]
            ax.fill(*zip(*polygon), alpha=0.6, color="lightgrey")
            ax.fill(*zip(*polygon), alpha=0.6, edgecolor="black", fill=False)

        force_vector = np.sum(force_matrix[basic_index, cells[basic_index].neighbours], axis=0)
        ax.quiver(cells[basic_index].x, cells[basic_index].y, force_vector[0], force_vector[1])

    for multiciliated_index in multiciliated_indices:
        region = voronoi.regions[voronoi.point_region[multiciliated_index]]
        if -1 not in region:
            polygon = voronoi.vertices[region]
            ax.fill(*zip(*polygon), alpha=0.6, color="orange")
            ax.fill(*zip(*polygon), alpha=0.6, edgecolor="black", fill=False)

        force_vector = np.sum(force_matrix[multiciliated_index, cells[multiciliated_index].neighbours], axis=0)
        ax.quiver(cells[multiciliated_index].x, cells[multiciliated_index].y, force_vector[0], force_vector[1])

    if boundary_points:
        boundary_array = np.array(boundary_points)
        ax.scatter(boundary_array[:, 0], boundary_array[:, 1], s=20, color="green")   

    fig.set_figwidth(8)
    fig.set_figheight(8)
    
    if information:
        information_box = dict(boxstyle="round", facecolor="white")
        fig.text(0.05, 0.95, information, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=information_box)

    if x_lim and y_lim:
        plt.xlim([0, x_lim])
        plt.ylim([0, y_lim])

    plt.title(title)
    plt.show()
    plt.pause(duration)

    ax.clear()
    
    return (fig, ax, None)

def plot_major_axes(cells: list, title: str, duration: float, plot: tuple = (None, None), x_lim: int = 0, y_lim: int = 0, information: str = ""):
    points = []
    basic_points = []
    basic_indices = []
    multiciliated_points = []
    multiciliated_indices = []
    boundary_points = []
    boundary_indices = []

    for i in range(len(cells)):
        points.append([cells[i].x, cells[i].y])
        if isinstance(cells[i], BasicCell):
            basic_points.append([cells[i].x, cells[i].y])
            basic_indices.append(i)
        elif isinstance(cells[i], BorderCell):
            boundary_points.append([cells[i].x, cells[i].y])
            boundary_indices.append(i)
        else:
            multiciliated_points.append([cells[i].x, cells[i].y])
            multiciliated_indices.append(i)

    voronoi = Voronoi(points)

    if not plot[0] and not plot[1]:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot[0], plot[1]
        
    for basic_index in basic_indices:
        region = voronoi.regions[voronoi.point_region[basic_index]]
        if -1 not in region:
            polygon = voronoi.vertices[region]
            ax.fill(*zip(*polygon), alpha=0.6, color="lightgrey")
            ax.fill(*zip(*polygon), alpha=0.6, edgecolor="black", fill=False)

            center = np.array([cells[basic_index].x, cells[basic_index].y])
            centered_points = polygon - center 
            covariance_matrix = np.cov(centered_points, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
            major_axis_index = np.argmax(eigenvalues)
            major_axis = eigenvectors[:, major_axis_index]
            
            projections = np.dot(centered_points, major_axis)
            length = projections.max() - projections.min()

            positive_end = center + (length / 2) * major_axis
            negative_end = center - (length / 2) * major_axis

            ax.plot([negative_end[0], positive_end[0]], [negative_end[1], positive_end[1]], linewidth=2, color="red")

    for multiciliated_index in multiciliated_indices:
        region = voronoi.regions[voronoi.point_region[multiciliated_index]]
        if -1 not in region:
            polygon = voronoi.vertices[region]
            ax.fill(*zip(*polygon), alpha=0.6, color="orange")
            ax.fill(*zip(*polygon), alpha=0.6, edgecolor="black", fill=False)

            center = np.array([cells[multiciliated_index].x, cells[multiciliated_index].y])
            centered_points = polygon - center 
            covariance_matrix = np.cov(centered_points, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
            major_axis_index = np.argmax(eigenvalues)
            major_axis = eigenvectors[:, major_axis_index]
            
            projections = np.dot(centered_points, major_axis)
            length = projections.max() - projections.min()

            positive_end = center + (length / 2) * major_axis
            negative_end = center - (length / 2) * major_axis

            ax.plot([negative_end[0], positive_end[0]], [negative_end[1], positive_end[1]], linewidth=2, color="red")

    if boundary_points:
        boundary_array = np.array(boundary_points)
        ax.scatter(boundary_array[:, 0], boundary_array[:, 1], s=20, color="green")   

    fig.set_figwidth(8)
    fig.set_figheight(8)
    
    if information:
        information_box = dict(boxstyle="round", facecolor="white")
        fig.text(0.05, 0.95, information, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=information_box)

    if x_lim and y_lim:
        plt.xlim([0, x_lim])
        plt.ylim([0, y_lim])

    plt.title(title)
    plt.show()
    plt.pause(duration)

    ax.clear()
    
    return (fig, ax, None)


