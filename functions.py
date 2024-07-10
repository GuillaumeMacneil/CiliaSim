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

