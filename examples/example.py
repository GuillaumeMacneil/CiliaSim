import numpy as np
from scipy.spatial import Voronoi

# Generate random points
np.random.seed(42)
points = np.random.rand(15, 2)  # 15 random points in 2D

# Create Voronoi diagram
vor = Voronoi(points)

print(vor.regions)
print(vor.ridge_points)


# # Plot the Voronoi diagram
# voronoi_plot_2d(vor, show_vertices=True)

# # Plot the original points
# plt.plot(points[:, 0], points[:, 1], 'ko', label='Input points')

# # Customize the plot
# plt.title('Voronoi Diagram Example')
# plt.xlabel('X axis')
# plt.ylabel('Y axis')
# plt.grid(True)
# plt.legend()

# # Show the plot
# plt.show()

# # Example of accessing Voronoi diagram properties
# print("\nVoronoi diagram properties:")
# print(f"Number of points: {len(vor.points)}")
# print(f"Number of vertices: {len(vor.vertices)}")
# print(f"Number of regions: {len(vor.regions)}")

# # Example of getting the region for a specific point
# point_index = 0
# region_index = vor.point_region[point_index]
# region_vertices = vor.regions[region_index]
# print(f"\nVertices for region of point {point_index}:")
# print(region_vertices)
