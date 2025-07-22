import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection

def get_voronoi_polygons(cell_points, circumcenters, triangles, num_cells):
    polygons = []
    for cell_index in range(num_cells):
        mask = np.any(triangles == cell_index, axis=1)
        vertices = circumcenters[mask]
        relative = vertices - cell_points[cell_index]
        angles = np.arctan2(relative[:, 1], relative[:, 0])
        sorted_indices = np.argsort(angles)
        polygon = vertices[sorted_indices]
        polygons.append(polygon) 

    return polygons

def basic_animation(state_files, params):
    # Load the initial points and set up the plot
    initial_data = np.load(state_files[0])
    cell_points = initial_data["cell_points"]
    circumcenters = initial_data["circumcenters"]
    triangles = initial_data["triangles"]
    cell_types = initial_data["cell_types"]
    num_cells = np.sum(cell_types != -1)
    polygons = get_voronoi_polygons(cell_points, circumcenters, triangles, num_cells)

    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    patches = []
    for i in range(num_cells):
        polygon = polygons[i]
        colour = "lightgray"
        if cell_types[i] == 2:
            colour = "orange"
        elif cell_types[i] == 1:
            continue

        patch = Polygon(polygon, closed=True, facecolor=colour, edgecolor='black', linewidth=0.5)
        ax.add_patch(patch)
        patches.append(patch)

    boundary_mask = cell_types == 1
    boundary_points = cell_points[boundary_mask]
    boundary_scatter = ax.scatter(boundary_points[:, 0], boundary_points[:, 1], color="green", s=10)

    # Set the axes such that aspect ratio doesn't change jarringly
    ax.set_xlim(np.min(cell_points[:, 0]) - 1, np.max(cell_points[:, 0]) + 1)
    ax.set_ylim(np.min(cell_points[:, 1]) - 1, np.max(cell_points[:, 1]) + 1)   

    def update(frame):
        # Load the data required for a basic voronoi plot
        data = np.load(state_files[frame])
        cell_points = data["cell_points"]
        circumcenters = data["circumcenters"]
        triangles = data["triangles"]
        cell_types = data["cell_types"]
        num_cells = np.sum(cell_types != -1)

        polygons = get_voronoi_polygons(cell_points, circumcenters, triangles, num_cells)

        for patch, polygon in zip(patches, polygons):
            patch.set_xy(polygon)
            #patch.set_facecolor(color_map_func(cell_types[i]))

        boundary_mask = cell_types == 1
        boundary_points = cell_points[boundary_mask]
        boundary_scatter.set_offsets(boundary_points)

        return patches + [boundary_scatter]
       
    return FuncAnimation(fig, update, frames=len(state_files), interval=100, blit=True)


def spring_animation(state_files, params):
    # Load the initial points and set up the plot
    initial_data = np.load(state_files[0])
    cell_points = initial_data["cell_points"]
    circumcenters = initial_data["circumcenters"]
    triangles = initial_data["triangles"]
    cell_types = initial_data["cell_types"]
    num_cells = np.sum(cell_types != -1)
    polygons = get_voronoi_polygons(cell_points, circumcenters, triangles, num_cells)

    # Set up plot configuration
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(np.min(cell_points[:, 0]) - 1, np.max(cell_points[:, 0]) + 1)
    ax.set_ylim(np.min(cell_points[:, 1]) - 1, np.max(cell_points[:, 1]) + 1)   
    
    # Draw cell polygon patches
    patches = []
    for i in range(num_cells):
        polygon = polygons[i]
        colour = "lightgray"
        if cell_types[i] == 2:
            colour = "orange"
        elif cell_types[i] == 1:
            continue

        patch = Polygon(polygon, closed=True, facecolor=colour, edgecolor='black', linewidth=0.5)
        ax.add_patch(patch)
        patches.append(patch)

    # Draw boundary points
    boundary_mask = cell_types == 1
    boundary_points = cell_points[boundary_mask]
    boundary_scatter = ax.scatter(boundary_points[:, 0], boundary_points[:, 1], color="green", s=10)

    # Draw springs using triangle data
    springs = []
    for triangle in triangles:
        if np.any(triangle == -1):
            continue

        a, b, c = triangle
        springs.extend([
            [cell_points[a], cell_points[b]],
            [cell_points[b], cell_points[c]],
            [cell_points[c], cell_points[a]]
            ])
    
    springs_line_collection = LineCollection(springs, colors="gray", linewidths=0.5)
    ax.add_collection(springs_line_collection)

    def update(frame):
        # Load the data required for a basic voronoi plot
        data = np.load(state_files[frame])
        cell_points = data["cell_points"]
        circumcenters = data["circumcenters"]
        triangles = data["triangles"]
        cell_types = data["cell_types"]
        num_cells = np.sum(cell_types != -1)

        # Update cell polygon patches
        polygons = get_voronoi_polygons(cell_points, circumcenters, triangles, num_cells)
        for patch, polygon in zip(patches, polygons):
            patch.set_xy(polygon)

        # Update boundary points
        boundary_mask = cell_types == 1
        boundary_points = cell_points[boundary_mask]
        boundary_scatter.set_offsets(boundary_points)

        # Update springs using triangle data
        springs = []
        for triangle in triangles:
            if np.any(triangle == -1):
                continue

            a, b, c = triangle
            springs.extend([
                [cell_points[a], cell_points[b]],
                [cell_points[b], cell_points[c]],
                [cell_points[c], cell_points[a]]
                ])
        
        springs_line_collection.set_segments(springs)

        return patches + [boundary_scatter, springs_line_collection]
       
    return FuncAnimation(fig, update, frames=len(state_files), interval=100, blit=True)

