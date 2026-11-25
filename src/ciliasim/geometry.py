import jax.numpy as jnp


def calculate_circumcenters(cell_points, triangles):
    """
    Calculate the circumcenter of each triangle.

    Args:
        cell_points: Cartesian positions of all cells.
        triangles: Padded array of triangle vertex indices.

    Returns:
        Circumcenter position for each triangle, with zero for invalid triangles.
    """
    valid = jnp.all(triangles != -1, axis=1)
    safe_triangles = jnp.where(valid[:, None], triangles, 0)

    A = cell_points[safe_triangles[:, 0]]
    B = cell_points[safe_triangles[:, 1]]
    C = cell_points[safe_triangles[:, 2]]

    a2 = jnp.sum(A**2, axis=1)
    b2 = jnp.sum(B**2, axis=1)
    c2 = jnp.sum(C**2, axis=1)

    ax, ay = A[:, 0], A[:, 1]
    bx, by = B[:, 0], B[:, 1]
    cx, cy = C[:, 0], C[:, 1]

    # The circumcenter denominator approaches zero for collinear triangles.
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

    edge1 = B - A
    edge2 = C - B
    edge3 = A - C
    max_edge = jnp.maximum(
        jnp.maximum(jnp.linalg.norm(edge1, axis=1), jnp.linalg.norm(edge2, axis=1)),
        jnp.linalg.norm(edge3, axis=1),
    )

    # Scale the degeneracy tolerance with triangle size.
    eps = 1e-8 * jnp.maximum(1.0, max_edge**2)
    degenerate = jnp.abs(d) < eps
    safe_d = jnp.where(degenerate, 1.0, d)

    ux_num = a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)
    uy_num = a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)

    centers = jnp.stack([ux_num / safe_d, uy_num / safe_d], axis=1)
    centroids = (A + B + C) / 3.0
    centers = jnp.where(degenerate[:, None], centroids, centers)
    centers = jnp.where(valid[:, None], centers, 0.0)

    return centers


def calculate_cell_areas(circumcenters, cell_triangles):
    """
    Calculate Voronoi cell areas from ordered triangle circumcenters.

    Args:
        circumcenters: Circumcenter position for each triangle.
        cell_triangles: Padded ordered incident-triangle indices for each cell.

    Returns:
        Area of each Voronoi cell.
    """
    valid = cell_triangles != -1
    safe_cell_triangles = jnp.where(valid, cell_triangles, 0)
    vertices = circumcenters[safe_cell_triangles]

    # Wrap the final valid vertex back to the first to close each polygon.
    num_vertices = jnp.sum(valid, axis=1)
    positions = jnp.arange(cell_triangles.shape[1])[None, :]
    next_positions = jnp.where(positions + 1 < num_vertices[:, None], positions + 1, 0)
    next_vertices = jnp.take_along_axis(vertices, next_positions[..., None], axis=1)

    # Apply the shoelace formula to each Voronoi polygon.
    cross_products = (
        vertices[..., 0] * next_vertices[..., 1]
        - vertices[..., 1] * next_vertices[..., 0]
    )
    cross_products = jnp.where(valid, cross_products, 0.0)

    return 0.5 * jnp.abs(jnp.sum(cross_products, axis=1))


def calculate_cell_perimeters(circumcenters, cell_triangles):
    """
    Calculate Voronoi cell perimeters from ordered triangle circumcenters.

    Args:
        circumcenters: Circumcenter position for each triangle.
        cell_triangles: Padded ordered incident-triangle indices for each cell.

    Returns:
        Perimeter of each Voronoi cell.
    """
    valid = cell_triangles != -1
    safe_cell_triangles = jnp.where(valid, cell_triangles, 0)
    vertices = circumcenters[safe_cell_triangles]

    # Wrap the final valid vertex back to the first to close each polygon.
    num_vertices = jnp.sum(valid, axis=1)
    positions = jnp.arange(cell_triangles.shape[1])[None, :]
    next_positions = jnp.where(positions + 1 < num_vertices[:, None], positions + 1, 0)
    next_vertices = jnp.take_along_axis(vertices, next_positions[..., None], axis=1)

    edge_vectors = next_vertices - vertices
    squared_lengths = jnp.sum(edge_vectors**2, axis=-1)

    # Regularization prevents a singular gradient for zero-length Voronoi edges.
    edge_lengths = jnp.sqrt(squared_lengths + 1e-12)
    edge_lengths = jnp.where(valid, edge_lengths, 0.0)

    return jnp.sum(edge_lengths, axis=1)
