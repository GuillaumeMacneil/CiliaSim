import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

def get_voronoi_edges(cell_points, circumcenters, triangles, num_cells):
    edges = []
    for cell_index in range(num_cells):
        mask = np.any(triangles == cell_index, axis=1)
        vertices = circumcenters[mask]
        relative = vertices - cell_points[cell_index]
        angles = np.arctan2(relative[:, 1], relative[:, 0])
        sorted_indices = np.argsort(angles)
        polygon = vertices[sorted_indices]

        edges += [
            [polygon[i], polygon[(i + 1) % len(polygon)]]
            for i in range(len(polygon))
        ]

    return edges

def basic_animation(state_files, params):
    # Load the initial points and set up the plot
    initial_data = np.load(state_files[0])
    cell_points = initial_data["cell_points"]
    circumcenters = initial_data["circumcenters"]
    triangles = initial_data["triangles"]
    cell_types = initial_data["cell_types"]
    num_cells = np.sum(cell_types != -1)
    edges = get_voronoi_edges(cell_points, circumcenters, triangles, num_cells)

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    line_collection = LineCollection(edges, colors="blue", linewidths=1, alpha=0.6)
    ax.add_collection(line_collection)

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

        edges = get_voronoi_edges(cell_points, circumcenters, triangles, num_cells)
        line_collection.set_segments(edges)
        return line_collection,
       
    return FuncAnimation(fig, update, frames=len(state_files), interval=100, blit=True)

