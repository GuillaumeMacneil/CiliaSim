import numpy as np

def polygon_area(vertices: np.ndarray):
    """
    Calculates the area of a polygon given its vertices using the shoelace formula.

    Args:
        vertices (np.ndarray): The vertices of the polygon.

    Returns:
        float: The area of the polygon.
    """
    x = vertices[:, 0]
    y = vertices[:, 1]

    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def polygon_perimeter(vertices: np.ndarray):
    looped_vertices = np.append(vertices, [vertices[0]], axis=0)
    differences = np.diff(looped_vertices, axis=0)
    distances = np.sqrt((differences ** 2).sum(axis=1))

    return np.sum(distances)
