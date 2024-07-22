from functions import *

import matplotlib.pyplot as plt
import matplotlib.colors as mcolours
import matplotlib.cm as cm
import numpy as np
from scipy.spatial import Voronoi, Delaunay

def plot_tissue(points: np.ndarray, cell_types: np.ndarray, title: str, duration: float, plot: tuple = (None, None), x_lim: int = 0, y_lim: int = 0, information: str = "", auto: bool = True):
    basic_indices = np.where(cell_types == 0)[0]
    boundary_indices = np.where(cell_types == 1)[0]
    boundary_points = points[boundary_indices]
    multiciliated_indices = np.where(cell_types == 2)[0]

    voronoi = Voronoi(points)

    if not plot[0] and not plot[1]:
        fig, ax = plt.subplots()
        information_box = None
    else:
        fig, ax, information_box = plot[0], plot[1], plot[3]
        ax.clear()

    for basic_index in basic_indices:
        region = voronoi.regions[voronoi.point_region[basic_index]]
        if -1 not in region:
            polygon = voronoi.vertices[region]
            ax.fill(*zip(*polygon), alpha=0.6, color="lightgrey", edgecolor="black")

    for multiciliated_index in multiciliated_indices:
        region = voronoi.regions[voronoi.point_region[multiciliated_index]]
        if -1 not in region:
            polygon = voronoi.vertices[region]
            ax.fill(*zip(*polygon), alpha=0.6, color="orange", edgecolor="black")

    ax.scatter(boundary_points[:, 0], boundary_points[:, 1], s=20, color="green")   

    fig.set_figwidth(8)
    fig.set_figheight(8)
    
    if not information_box:
        information_dict = dict(boxstyle="round", facecolor="white", alpha=0.5)
        information_box = fig.text(0.05, 0.95, information, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=information_dict)
    else:
        information_box.set_text(information)

    if x_lim and y_lim:
        plt.xlim([0, x_lim])
        plt.ylim([0, y_lim])

    plt.title(title)

    if auto:
        plt.show()
        plt.pause(duration)
    
    return (fig, ax, None, information_box)

def plot_springs(points: np.ndarray, cell_types: np.ndarray, adjacency_matrix: np.ndarray, title: str, duration: float, plot: tuple = (None, None), x_lim: int = 0, y_lim: int = 0, information: str = "", auto: bool = True):
    basic_indices = np.where(cell_types == 0)[0]
    boundary_indices = np.where(cell_types == 1)[0]
    boundary_points = points[boundary_indices]
    multiciliated_indices = np.where(cell_types == 2)[0]

    voronoi = Voronoi(points)

    if not plot[0] and not plot[1]:
        fig, ax = plt.subplots()
        information_box = None
    else:
        fig, ax, information_box = plot[0], plot[1], plot[3]
        ax.clear()

    if not plot[2]:
        colourbar = None
    else:
        colourbar = plot[2]
        
    for basic_index in basic_indices:
        region = voronoi.regions[voronoi.point_region[basic_index]]
        if -1 not in region:
            polygon = voronoi.vertices[region]
            ax.fill(*zip(*polygon), alpha=0.6, color="lightgrey", edgecolor="black")

    for multiciliated_index in multiciliated_indices:
        region = voronoi.regions[voronoi.point_region[multiciliated_index]]
        if -1 not in region:
            polygon = voronoi.vertices[region]
            ax.fill(*zip(*polygon), alpha=0.6, color="orange", edgecolor="black")

    ax.scatter(boundary_points[:, 0], boundary_points[:, 1], s=20, color="green")   

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
            ax.plot([points[i, 0], neighbour_points[j, 0]], [points[i, 1], neighbour_points[j, 1]], color=colour_matrix[i][neighbours][j])

    fig.set_figwidth(10)
    fig.set_figheight(8)
    
    if not information_box:
        information_dict = dict(boxstyle="round", facecolor="white", alpha=0.5)
        information_box = fig.text(0.05, 0.95, information, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=information_dict)
    else:
        information_box.set_text(information)

    if not colourbar:
        colourbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colourmap), ax=ax, orientation="vertical")
    else:
        colourbar.update_normal(cm.ScalarMappable(norm=norm, cmap=colourmap))

    if x_lim and y_lim:
        plt.xlim([0, x_lim])
        plt.ylim([0, y_lim])

    plt.title(title)

    if auto:
        plt.show()
        plt.pause(duration)

    return (fig, ax, colourbar, information_box)

def plot_force_vectors(points: np.ndarray, cell_types: np.ndarray, force_matrix: np.ndarray, title: str, duration: float, plot: tuple = (None, None), x_lim: int = 0, y_lim: int = 0, information: str = "", auto: bool = True):
    basic_indices = np.where(cell_types == 0)[0]
    boundary_indices = np.where(cell_types == 1)[0]
    boundary_points = points[boundary_indices]
    multiciliated_indices = np.where(cell_types == 2)[0]

    voronoi = Voronoi(points)

    if not plot[0] and not plot[1]:
        fig, ax = plt.subplots()
        information_box = None
    else:
        fig, ax, information_box = plot[0], plot[1], plot[3]
        ax.clear()

    for basic_index in basic_indices:
        region = voronoi.regions[voronoi.point_region[basic_index]]
        if -1 not in region:
            polygon = voronoi.vertices[region]
            ax.fill(*zip(*polygon), alpha=0.6, color="lightgrey", edgecolor="black")

    for multiciliated_index in multiciliated_indices:
        region = voronoi.regions[voronoi.point_region[multiciliated_index]]
        if -1 not in region:
            polygon = voronoi.vertices[region]
            ax.fill(*zip(*polygon), alpha=0.6, color="orange", edgecolor="black")

    ax.scatter(boundary_points[:, 0], boundary_points[:, 1], s=20, color="green")   

    force_vector = np.array([np.sum(force_matrix[point_index], axis=0) for point_index in range(len(points))])
    ax.quiver(points[:, 0], points[:, 1], force_vector[:, 0], force_vector[:, 1])

    fig.set_figwidth(8)
    fig.set_figheight(8)
    
    if not information_box:
        information_dict = dict(boxstyle="round", facecolor="white", alpha=0.5)
        information_box = fig.text(0.05, 0.95, information, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=information_dict)
    else:
        information_box.set_text(information)

    if x_lim and y_lim:
        plt.xlim([0, x_lim])
        plt.ylim([0, y_lim])

    plt.title(title)

    if auto:
        plt.show()
        plt.pause(duration)
 
    return (fig, ax, None, information_box)

def plot_major_axes(points: np.ndarray, cell_types: np.ndarray, title: str, duration: float, plot: tuple = (None, None), x_lim: int = 0, y_lim: int = 0, information: str = "", auto: bool = True):
    basic_indices = np.where(cell_types == 0)[0]
    boundary_indices = np.where(cell_types == 1)[0]
    boundary_points = points[boundary_indices]
    multiciliated_indices = np.where(cell_types == 2)[0]

    voronoi = Voronoi(points)

    if not plot[0] and not plot[1]:
        fig, ax = plt.subplots()
        information_box = None
    else:
        fig, ax, information_box = plot[0], plot[1], plot[3]    
        ax.clear()

    for basic_index in basic_indices:
        region = voronoi.regions[voronoi.point_region[basic_index]] 
        if -1 not in region:
            polygon = voronoi.vertices[region]
            ax.fill(*zip(*polygon), alpha=0.6, color="lightgrey", edgecolor="black")

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

            ax.plot([negative_end[0], positive_end[0]], [negative_end[1], positive_end[1]], linewidth=2, color="red")

    for multiciliated_index in multiciliated_indices:
        region = voronoi.regions[voronoi.point_region[multiciliated_index]]
        if -1 not in region:
            polygon = voronoi.vertices[region]
            ax.fill(*zip(*polygon), alpha=0.6, color="orange", edgecolor="black")

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

            ax.plot([negative_end[0], positive_end[0]], [negative_end[1], positive_end[1]], linewidth=2, color="red")

    ax.scatter(boundary_points[:, 0], boundary_points[:, 1], s=20, color="green")   

    fig.set_figwidth(8)
    fig.set_figheight(8)
    
    if not information_box:
        information_dict = dict(boxstyle="round", facecolor="white", alpha=0.5)
        information_box = fig.text(0.05, 0.95, information, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=information_dict)
    else:
        information_box.set_text(information)

    if x_lim and y_lim:
        plt.xlim([0, x_lim])
        plt.ylim([0, y_lim])

    plt.title(title)

    if auto:
        plt.show()
        plt.pause(duration)
    
    return (fig, ax, None, information_box)

def plot_avg_major_axes(points: np.ndarray, cell_types: np.ndarray, adjacency_matrix: np.ndarray, title: str, duration: float, plot: tuple = (None, None), x_lim: int = 0, y_lim: int = 0, information: str = "", auto: bool = True):
    basic_indices = np.where(cell_types == 0)[0]
    boundary_indices = np.where(cell_types == 1)[0]
    boundary_points = points[boundary_indices]
    multiciliated_indices = np.where(cell_types == 2)[0]

    voronoi = Voronoi(points)

    if not plot[0] and not plot[1]:
        fig, ax = plt.subplots()
        information_box = None
    else:
        fig, ax, information_box = plot[0], plot[1], plot[3]    
        ax.clear()
       
    for basic_index in basic_indices:
        region = voronoi.regions[voronoi.point_region[basic_index]]
        if -1 not in region:
            polygon = voronoi.vertices[region]
            ax.fill(*zip(*polygon), alpha=0.6, color="lightgrey", edgecolor="black")

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

            ax.plot([negative_end[0], positive_end[0]], [negative_end[1], positive_end[1]], linewidth=2, color="red")

    for multiciliated_index in multiciliated_indices:
        region = voronoi.regions[voronoi.point_region[multiciliated_index]]
        if -1 not in region:
            polygon = voronoi.vertices[region]
            ax.fill(*zip(*polygon), alpha=0.6, color="lightgrey", edgecolor="black")

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

            ax.plot([negative_end[0], positive_end[0]], [negative_end[1], positive_end[1]], linewidth=2, color="red")

    ax.scatter(boundary_points[:, 0], boundary_points[:, 1], s=20, color="green")   

    fig.set_figwidth(8)
    fig.set_figheight(8)
    
    if not information_box:
        information_dict = dict(boxstyle="round", facecolor="white", alpha=0.5)
        information_box = fig.text(0.05, 0.95, information, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=information_dict)
    else:
        information_box.set_text(information)

    if x_lim and y_lim:
        plt.xlim([0, x_lim])
        plt.ylim([0, y_lim])

    plt.title(title)
    
    if auto:
        plt.show()
        plt.pause(duration)
    
    return (fig, ax, None, information_box)

def plot_area_delta(points: np.ndarray, cell_types: np.ndarray, target_area: float, title: str, duration: float, plot: tuple = (None, None), information: str = "", auto: bool = True):
    voronoi = Voronoi(points)

    area_diffs = []
    non_border_indices = np.where(cell_types != 1)[0]
    for i in non_border_indices:
        area = polygon_area(voronoi.vertices[voronoi.regions[voronoi.point_region[i]]]) 
        area_diffs.append(area - target_area)

    if not plot[0] and not plot[1]:
        fig, ax = plt.subplots()
        information_box = None
    else:
        fig, ax, information_box = plot[0], plot[1], plot[3]
        ax.clear()

    ax.hist(area_diffs)

    fig.set_figwidth(8)
    fig.set_figheight(8)
    
    if not information_box:
        information_dict = dict(boxstyle="round", facecolor="white", alpha=0.5)
        information_box = fig.text(0.05, 0.95, information, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=information_dict)
    else:
        information_box.set_text(information)

    plt.title(title)

    if auto:
        plt.show()
        plt.pause(duration)
    
    return (fig, ax, None, information_box)

def plot_neighbour_histogram(adjacency_matrix: np.ndarray, title: str, duration: float, plot: tuple = (None, None), information: str = "", auto: bool = True):
    connectivity = np.sum(adjacency_matrix, axis=0)
    disc_size = np.diff(np.unique(connectivity)).min()
    left = connectivity.min() - float(disc_size) / 2
    right = connectivity.max() + float(disc_size) / 2

    if not plot[0] and not plot[1]:
        fig, ax = plt.subplots()
        information_box = None
    else:
        fig, ax, information_box = plot[0], plot[1], plot[3]
        ax.clear()

    ax.hist(connectivity, np.arange(left, right + disc_size, disc_size))

    fig.set_figwidth(8)
    fig.set_figheight(8)
    
    if not information_box:
        information_dict = dict(boxstyle="round", facecolor="white", alpha=0.5)
        information_box = fig.text(0.05, 0.95, information, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=information_dict)
    else:
        information_box.set_text(information)

    plt.title(title)

    if auto:
        plt.show()
        plt.pause(duration)
    
    return (fig, ax, None, information_box)

def plot_shape_factor_histogram(shape_factors: np.ndarray, title: str, duration: float, plot: tuple = (None, None), information: str = "", auto: bool = True):
    if not plot[0] and not plot[1]:
        fig, ax = plt.subplots()
        information_box = None
    else:
        fig, ax, information_box = plot[0], plot[1], plot[3]
        ax.clear()

    ax.hist(shape_factors)

    fig.set_figwidth(8)
    fig.set_figheight(8)
    
    if not information_box:
        information_dict = dict(boxstyle="round", facecolor="white", alpha=0.5)
        information_box = fig.text(0.05, 0.95, information, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=information_dict)
    else:
        information_box.set_text(information)

    plt.title(title)
    
    if auto:
        plt.show()
        plt.pause(duration)
    
    return (fig, ax, None, information_box)

