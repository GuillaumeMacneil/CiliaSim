import numpy as np

def polygon_area(vertices: np.ndarray):
    x = vertices[:, 0]
    y = vertices[:, 1]

    sum_a = np.dot(x, y[np.arange(len(y)) - 1])
    sum_b = np.dot(y, x[np.arange(len(x)) - 1])

    return 0.5 * np.abs(sum_a - sum_b)

def polygon_perimeter(vertices: np.ndarray):
    looped_vertices = np.append(vertices, [vertices[0]], axis=0)
    differences = np.diff(looped_vertices, axis=0)
    distances = np.sqrt((differences ** 2).sum(axis=1))

    return np.sum(distances)
