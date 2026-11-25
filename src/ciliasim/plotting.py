import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PolyCollection
from scipy.interpolate import griddata


def load_state(state_file):
    """
    Load all arrays from a saved simulation state.

    Args:
        state_file: Path to the saved NPZ state.

    Returns:
        Dictionary mapping saved field names to arrays.
    """
    with np.load(state_file) as data:
        return {key: data[key] for key in data.files}


def get_real_mask(cell_types):
    """
    Identify non-boundary, non-vacant cells.

    Args:
        cell_types: Cell type array.

    Returns:
        Boolean mask identifying real tissue cells.
    """
    return (cell_types != 1) & (cell_types != -1)


def get_active_mask(cell_types):
    """
    Identify all active cells, including boundary cells.

    Args:
        cell_types: Cell type array.

    Returns:
        Boolean mask identifying non-vacant cells.
    """
    return cell_types != -1


def get_plot_limits(state_files, padding=1.0):
    """
    Determine fixed plot limits spanning all simulation states.

    Args:
        state_files: Ordered collection of saved state paths.
        padding: Additional space added around the tissue bounds.

    Returns:
        Minimum and maximum x and y plot coordinates.
    """
    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf

    for state_file in state_files:
        with np.load(state_file) as data:
            cell_points = data["cell_points"]
            cell_types = data["cell_types"]

            # Include boundary cells so the complete tissue remains visible.
            active_points = cell_points[get_active_mask(cell_types)]

            x_min = min(x_min, np.min(active_points[:, 0]))
            x_max = max(x_max, np.max(active_points[:, 0]))
            y_min = min(y_min, np.min(active_points[:, 1]))
            y_max = max(y_max, np.max(active_points[:, 1]))

    return x_min - padding, x_max + padding, y_min - padding, y_max + padding


def get_real_polygons(cell_types, circumcenters, cell_triangles):
    """
    Construct Voronoi polygons and display colours for real cells.

    Args:
        cell_types: Cell type array.
        circumcenters: Circumcenter position for each triangle.
        cell_triangles: Ordered incident-triangle indices for each cell.

    Returns:
        Voronoi polygons and their corresponding face colours.
    """
    real_indices = np.where(get_real_mask(cell_types))[0]

    polygons = []
    colours = []

    for cell in real_indices:
        triangle_indices = cell_triangles[cell]
        triangle_indices = triangle_indices[triangle_indices >= 0]

        if len(triangle_indices) < 3:
            continue

        # Ordered circumcenters form the vertices of the cell's Voronoi polygon.
        polygons.append(circumcenters[triangle_indices])
        colours.append("orange" if cell_types[cell] == 2 else "lightgray")

    return polygons, colours


def get_spring_edges(adjacency):
    """
    Extract unique spring edges from symmetric cell adjacency.

    Args:
        adjacency: Padded cell adjacency array.

    Returns:
        Array of unique cell-index pairs connected by springs.
    """
    rows, slots = np.where(adjacency >= 0)
    cols = adjacency[rows, slots]

    # Keep each symmetric adjacency pair only once.
    keep = rows < cols

    return np.column_stack((rows[keep], cols[keep]))


def get_spring_segments(cell_points, adjacency):
    """
    Construct line segments for all unique springs.

    Args:
        cell_points: Cell positions.
        adjacency: Padded cell adjacency array.

    Returns:
        Spring line segments and their corresponding cell-index pairs.
    """
    edges = get_spring_edges(adjacency)

    if len(edges) == 0:
        return np.empty((0, 2, 2)), edges

    segments = np.stack((cell_points[edges[:, 0]], cell_points[edges[:, 1]]), axis=1)

    return segments, edges


def get_major_axis_segments(polygons):
    """
    Calculate major-axis line segments for a collection of polygons.

    Args:
        polygons: Collection of Voronoi polygon vertices.

    Returns:
        Array of major-axis line segments.
    """
    segments = []

    for polygon in polygons:
        coords = compute_major_axis(polygon)

        if coords is None:
            continue

        x1, y1, x2, y2 = coords
        segments.append(((x1, y1), (x2, y2)))

    return np.asarray(segments)


def get_spring_forces(cell_points, edges, spring_length, spring_constant):
    """
    Calculate signed spring forces from cell separation.

    Args:
        cell_points: Cell positions.
        edges: Cell-index pairs defining springs.
        spring_length: Preferred spring length.
        spring_constant: Spring stiffness.

    Returns:
        Signed force magnitude for each spring.
    """
    if len(edges) == 0:
        return np.empty(0)

    differences = cell_points[edges[:, 1]] - cell_points[edges[:, 0]]
    distances = np.linalg.norm(differences, axis=1)

    # Positive values indicate extension and negative values compression.
    return spring_constant * (distances - spring_length)


def get_area_strains(cell_types, target_areas, circumcenters, cell_triangles):
    """
    Calculate relative area strain for each real Voronoi cell.

    Args:
        cell_types: Cell type array.
        target_areas: Preferred area of each cell.
        circumcenters: Circumcenter position for each triangle.
        cell_triangles: Ordered incident-triangle indices for each cell.

    Returns:
        Voronoi polygons and their corresponding relative area strains.
    """
    polygons = []
    strains = []

    for cell in np.where(get_real_mask(cell_types))[0]:
        triangle_indices = cell_triangles[cell]
        triangle_indices = triangle_indices[triangle_indices >= 0]

        if len(triangle_indices) < 3 or target_areas[cell] <= 0:
            continue

        polygon = circumcenters[triangle_indices]
        x = polygon[:, 0]
        y = polygon[:, 1]

        # Calculate polygon area using the shoelace formula.
        area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))

        polygons.append(polygon)
        strains.append((area - target_areas[cell]) / target_areas[cell])

    return polygons, np.asarray(strains)


def get_reporter_text(data):
    """
    Format summary statistics for an animation frame.

    Args:
        data: Loaded simulation state containing forces, cell types, and iteration.

    Returns:
        Formatted reporter text for display over the animation.
    """
    cell_types = data["cell_types"]
    internal_force = data["internal_force"]
    cilia_force = data["cilia_force"]
    flow_force = data["flow_force"]

    real_mask = get_real_mask(cell_types)
    mcc_mask = cell_types == 2

    internal_magnitudes = np.linalg.norm(internal_force[real_mask], axis=1)
    cilia_magnitudes = np.linalg.norm(cilia_force[mcc_mask], axis=1)

    iteration = int(data["iteration"])
    num_cells = np.sum(real_mask)
    num_mcc = np.sum(mcc_mask)
    num_boundary = np.sum(cell_types == 1)

    mean_internal = np.mean(internal_magnitudes) if len(internal_magnitudes) > 0 else 0.0
    max_internal = np.max(internal_magnitudes) if len(internal_magnitudes) > 0 else 0.0

    mean_cilia = np.mean(cilia_magnitudes) if len(cilia_magnitudes) > 0 else 0.0
    max_cilia = np.max(cilia_magnitudes) if len(cilia_magnitudes) > 0 else 0.0

    flow_magnitude = np.linalg.norm(flow_force)

    return (
        f"Iteration: {iteration:,}    "
        f"Cells: {num_cells:,}    "
        f"MCCs: {num_mcc:,}    "
        f"Boundary: {num_boundary:,}\n"
        f"Internal |F|: {mean_internal:.3e} mean / {max_internal:.3e} max\n"
        f"Ciliary |F|: {mean_cilia:.3e} mean / {max_cilia:.3e} max    "
        f"Flow |F|: {flow_magnitude:.3e}"
    )


def add_reporter(ax, data):
    """
    Add a simulation-state reporter to an animation axes.

    Args:
        ax: Matplotlib axes on which to place the reporter.
        data: Initial simulation state used to populate the reporter.

    Returns:
        Matplotlib text artist containing the reporter.
    """
    return ax.text(
        0.02,
        0.98,
        get_reporter_text(data),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        family="monospace",
        bbox={
            "facecolor": "white",
            "edgecolor": "black",
            "alpha": 0.8,
            "pad": 4,
        },
        zorder=100,
    )


def basic_animation(state_files, params):
    """
    Animate Voronoi cell geometry over a simulation.

    Args:
        state_files: Ordered collection of saved simulation states.
        params: Static simulation parameters.

    Returns:
        Matplotlib animation of the tissue geometry.
    """
    initial_data = load_state(state_files[0])

    cell_points = initial_data["cell_points"]
    circumcenters = initial_data["circumcenters"]
    cell_triangles = initial_data["cell_triangles"]
    cell_types = initial_data["cell_types"]

    polygons, colours = get_real_polygons(cell_types, circumcenters, cell_triangles)
    x_min, x_max, y_min, y_max = get_plot_limits(state_files)

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    polygon_collection = PolyCollection(polygons, facecolors=colours, edgecolors="black", linewidths=0.5)
    ax.add_collection(polygon_collection)

    boundary_points = cell_points[cell_types == 1]
    boundary_scatter = ax.scatter(boundary_points[:, 0], boundary_points[:, 1], color="green", s=10)

    reporter = add_reporter(ax, initial_data)

    def update(frame):
        data = load_state(state_files[frame])

        cell_points = data["cell_points"]
        circumcenters = data["circumcenters"]
        cell_triangles = data["cell_triangles"]
        cell_types = data["cell_types"]

        polygons, colours = get_real_polygons(cell_types, circumcenters, cell_triangles)
        polygon_collection.set_verts(polygons)
        polygon_collection.set_facecolors(colours)

        boundary_points = cell_points[cell_types == 1]
        boundary_scatter.set_offsets(boundary_points)

        reporter.set_text(get_reporter_text(data))

        return polygon_collection, boundary_scatter, reporter

    return FuncAnimation(fig, update, frames=len(state_files), interval=100, blit=True)


def spring_animation(state_files, params):
    """
    Animate tissue geometry with springs coloured by signed force.

    Args:
        state_files: Ordered collection of saved simulation states.
        params: Static simulation parameters including spring properties.

    Returns:
        Matplotlib animation of the tissue and spring forces.
    """
    initial_data = load_state(state_files[0])

    cell_points = initial_data["cell_points"]
    circumcenters = initial_data["circumcenters"]
    cell_triangles = initial_data["cell_triangles"]
    cell_types = initial_data["cell_types"]
    adjacency = initial_data["adjacency"]

    spring_length = params["spring_length"]
    spring_constant = params["spring_constant"]

    polygons, colours = get_real_polygons(cell_types, circumcenters, cell_triangles)
    segments, edges = get_spring_segments(cell_points, adjacency)
    spring_forces = get_spring_forces(cell_points, edges, spring_length, spring_constant)

    x_min, x_max, y_min, y_max = get_plot_limits(state_files)

    max_force = 1e-12

    for state_file in state_files:
        with np.load(state_file) as data:
            _, frame_edges = get_spring_segments(data["cell_points"], data["adjacency"])
            frame_forces = get_spring_forces(data["cell_points"], frame_edges, spring_length, spring_constant)

            if len(frame_forces) > 0:
                max_force = max(max_force, np.max(np.abs(frame_forces)))

    norm = plt.Normalize(-max_force, max_force)

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    polygon_collection = PolyCollection(polygons, facecolors=colours, edgecolors="black", linewidths=0.5)
    ax.add_collection(polygon_collection)

    boundary_points = cell_points[cell_types == 1]
    boundary_scatter = ax.scatter(boundary_points[:, 0], boundary_points[:, 1], color="green", s=10)

    spring_collection = LineCollection(segments, cmap="coolwarm", norm=norm, linewidths=0.75)
    spring_collection.set_array(spring_forces)
    ax.add_collection(spring_collection)

    colorbar = fig.colorbar(spring_collection, ax=ax)
    colorbar.set_label("Spring force")

    reporter = add_reporter(ax, initial_data)

    def update(frame):
        data = load_state(state_files[frame])

        cell_points = data["cell_points"]
        circumcenters = data["circumcenters"]
        cell_triangles = data["cell_triangles"]
        cell_types = data["cell_types"]
        adjacency = data["adjacency"]

        polygons, colours = get_real_polygons(cell_types, circumcenters, cell_triangles)
        polygon_collection.set_verts(polygons)
        polygon_collection.set_facecolors(colours)

        boundary_points = cell_points[cell_types == 1]
        boundary_scatter.set_offsets(boundary_points)

        segments, edges = get_spring_segments(cell_points, adjacency)
        spring_forces = get_spring_forces(cell_points, edges, spring_length, spring_constant)

        spring_collection.set_segments(segments)
        spring_collection.set_array(spring_forces)

        reporter.set_text(get_reporter_text(data))

        return polygon_collection, boundary_scatter, spring_collection, reporter

    return FuncAnimation(fig, update, frames=len(state_files), interval=100, blit=True)


def compute_major_axis(polygon):
    """
    Calculate the major axis of a polygon using principal component analysis.

    Args:
        polygon: Ordered Cartesian vertices of the polygon.

    Returns:
        Major-axis endpoints as (x1, y1, x2, y2), or None if the axis cannot
        be determined.
    """
    vertices = np.asarray(polygon, dtype=np.float64)

    if len(vertices) < 2:
        return None

    centroid = np.mean(vertices, axis=0)
    centered_vertices = vertices - centroid

    if np.allclose(centered_vertices, 0):
        return None

    covariance = centered_vertices.T @ centered_vertices / len(vertices)

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return None

    major_direction = eigenvectors[:, np.argmax(eigenvalues)]
    direction_norm = np.linalg.norm(major_direction)

    if direction_norm == 0:
        return None

    major_direction /= direction_norm

    intersections = []

    # Intersect the infinite major-axis line with each polygon edge.
    for i in range(len(vertices)):
        edge_start = vertices[i]
        edge_end = vertices[(i + 1) % len(vertices)]
        edge_direction = edge_end - edge_start

        # Solve: edge_start + edge_fraction * edge_direction = centroid + axis_distance * major_direction
        system = np.column_stack((edge_direction, -major_direction))
        determinant = np.linalg.det(system)

        # Parallel lines have no unique intersection.
        if abs(determinant) < 1e-12:
            continue

        try:
            edge_fraction, axis_distance = np.linalg.solve(system, centroid - edge_start)
        except np.linalg.LinAlgError:
            continue

        # The intersection must lie on the finite polygon edge.
        if edge_fraction < -1e-9 or edge_fraction > 1.0 + 1e-9:
            continue

        intersection = centroid + axis_distance * major_direction

        # A polygon vertex may be found from both adjacent edges.
        if not any(np.allclose(intersection, existing, atol=1e-8) for existing in intersections):
            intersections.append(intersection)

    if len(intersections) >= 2:
        intersections = np.asarray(intersections)

        # Select the two boundary intersections furthest apart along the axis.
        axis_positions = (intersections - centroid) @ major_direction
        first_endpoint = intersections[np.argmin(axis_positions)]
        second_endpoint = intersections[np.argmax(axis_positions)]

        return first_endpoint[0], first_endpoint[1], second_endpoint[0], second_endpoint[1]

    # Project the polygon vertices onto the axis as a fallback.
    axis_positions = centered_vertices @ major_direction
    minimum_distance = np.min(axis_positions)
    maximum_distance = np.max(axis_positions)

    first_endpoint = centroid + minimum_distance * major_direction
    second_endpoint = centroid + maximum_distance * major_direction

    return first_endpoint[0], first_endpoint[1], second_endpoint[0], second_endpoint[1]


def major_axes_animation(state_files, params):
    """
    Animate Voronoi cells with their major axes overlaid.

    Args:
        state_files: Ordered collection of saved simulation states.
        params: Static simulation parameters.

    Returns:
        Matplotlib animation of tissue geometry and cell major axes.
    """
    initial_data = load_state(state_files[0])

    cell_points = initial_data["cell_points"]
    circumcenters = initial_data["circumcenters"]
    cell_triangles = initial_data["cell_triangles"]
    cell_types = initial_data["cell_types"]

    polygons, colours = get_real_polygons(cell_types, circumcenters, cell_triangles)
    axis_segments = get_major_axis_segments(polygons)

    x_min, x_max, y_min, y_max = get_plot_limits(state_files)

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    polygon_collection = PolyCollection(polygons, facecolors=colours, edgecolors="black", linewidths=0.5)
    ax.add_collection(polygon_collection)

    axis_collection = LineCollection(axis_segments, colors="red", linewidths=1.0)
    ax.add_collection(axis_collection)

    boundary_points = cell_points[cell_types == 1]
    boundary_scatter = ax.scatter(boundary_points[:, 0], boundary_points[:, 1], color="green", s=10)

    reporter = add_reporter(ax, initial_data)

    def update(frame):
        data = load_state(state_files[frame])

        cell_points = data["cell_points"]
        circumcenters = data["circumcenters"]
        cell_triangles = data["cell_triangles"]
        cell_types = data["cell_types"]

        polygons, colours = get_real_polygons(cell_types, circumcenters, cell_triangles)

        polygon_collection.set_verts(polygons)
        polygon_collection.set_facecolors(colours)

        axis_segments = get_major_axis_segments(polygons)
        axis_collection.set_segments(axis_segments)

        boundary_points = cell_points[cell_types == 1]
        boundary_scatter.set_offsets(boundary_points)

        reporter.set_text(get_reporter_text(data))

        return polygon_collection, axis_collection, boundary_scatter, reporter

    return FuncAnimation(fig, update, frames=len(state_files), interval=100, blit=True)


def boundary_animation(state_files, params):
    """
    Animate tissue geometry with the ordered boundary cycle highlighted.

    Args:
        state_files: Ordered collection of saved simulation states.
        params: Static simulation parameters.

    Returns:
        Matplotlib animation of the tissue and boundary cycle.
    """
    initial_data = load_state(state_files[0])

    cell_points = initial_data["cell_points"]
    circumcenters = initial_data["circumcenters"]
    cell_triangles = initial_data["cell_triangles"]
    cell_types = initial_data["cell_types"]
    boundary_cycle = initial_data["boundary_cycle"]

    polygons, colours = get_real_polygons(cell_types, circumcenters, cell_triangles)
    boundary_points = cell_points[boundary_cycle]
    closed_boundary = np.vstack((boundary_points, boundary_points[0]))

    x_min, x_max, y_min, y_max = get_plot_limits(state_files)

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    polygon_collection = PolyCollection(polygons, facecolors=colours, edgecolors="black", linewidths=0.5)
    ax.add_collection(polygon_collection)

    boundary_scatter = ax.scatter(boundary_points[:, 0], boundary_points[:, 1], color="green", s=10)
    boundary_line, = ax.plot(closed_boundary[:, 0], closed_boundary[:, 1], color="red", linewidth=1.5)

    reporter = add_reporter(ax, initial_data)

    def update(frame):
        data = load_state(state_files[frame])

        cell_points = data["cell_points"]
        circumcenters = data["circumcenters"]
        cell_triangles = data["cell_triangles"]
        cell_types = data["cell_types"]
        boundary_cycle = data["boundary_cycle"]

        polygons, colours = get_real_polygons(cell_types, circumcenters, cell_triangles)
        polygon_collection.set_verts(polygons)
        polygon_collection.set_facecolors(colours)

        boundary_points = cell_points[boundary_cycle]
        closed_boundary = np.vstack((boundary_points, boundary_points[0]))

        boundary_scatter.set_offsets(boundary_points)
        boundary_line.set_data(closed_boundary[:, 0], closed_boundary[:, 1])

        reporter.set_text(get_reporter_text(data))

        return polygon_collection, boundary_scatter, boundary_line, reporter

    return FuncAnimation(fig, update, frames=len(state_files), interval=100, blit=True)


def force_vectors_animation(state_files, params):
    """
    Animate cell forces as vectors over the Voronoi tissue.

    Args:
        state_files: Ordered collection of saved simulation states.
        params: Static simulation parameters.

    Returns:
        Matplotlib animation of tissue geometry and cell force vectors.
    """
    initial_data = load_state(state_files[0])

    cell_points = initial_data["cell_points"]
    circumcenters = initial_data["circumcenters"]
    cell_triangles = initial_data["cell_triangles"]
    cell_types = initial_data["cell_types"]
    total_force = initial_data["total_force"]

    x_min, x_max, y_min, y_max = get_plot_limits(state_files)

    max_force = 1e-12

    for state_file in state_files:
        with np.load(state_file) as data:
            real_mask = get_real_mask(data["cell_types"])
            forces = data["total_force"][real_mask]

            if len(forces) > 0:
                max_force = max(max_force, np.max(np.linalg.norm(forces, axis=1)))

    polygons, colours = get_real_polygons(cell_types, circumcenters, cell_triangles)

    real_mask = get_real_mask(cell_types)
    real_points = cell_points[real_mask]
    real_forces = total_force[real_mask] / max_force

    boundary_points = cell_points[cell_types == 1]

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    polygon_collection = PolyCollection(polygons, facecolors=colours, edgecolors="black", linewidths=0.5)
    ax.add_collection(polygon_collection)

    boundary_scatter = ax.scatter(boundary_points[:, 0], boundary_points[:, 1], color="green", s=10)

    quiver = ax.quiver(real_points[:, 0], real_points[:, 1], real_forces[:, 0], real_forces[:, 1], angles="xy", scale_units="xy", scale=1.5)

    reporter = add_reporter(ax, initial_data)

    def update(frame):
        data = load_state(state_files[frame])

        cell_points = data["cell_points"]
        circumcenters = data["circumcenters"]
        cell_triangles = data["cell_triangles"]
        cell_types = data["cell_types"]
        total_force = data["total_force"]

        polygons, colours = get_real_polygons(cell_types, circumcenters, cell_triangles)
        polygon_collection.set_verts(polygons)
        polygon_collection.set_facecolors(colours)

        real_mask = get_real_mask(cell_types)
        real_points = cell_points[real_mask]
        real_forces = total_force[real_mask] / max_force

        quiver.set_offsets(real_points)
        quiver.set_UVC(real_forces[:, 0], real_forces[:, 1])

        boundary_points = cell_points[cell_types == 1]
        boundary_scatter.set_offsets(boundary_points)

        reporter.set_text(get_reporter_text(data))

        return polygon_collection, boundary_scatter, quiver, reporter

    return FuncAnimation(fig, update, frames=len(state_files), interval=100, blit=True)


def force_field_animation(state_files, params, grid_size=50):
    """
    Animate an interpolated force field over the Voronoi tissue.

    Args:
        state_files: Ordered collection of saved simulation states.
        params: Static simulation parameters.
        grid_size: Number of interpolation points along each grid dimension.

    Returns:
        Matplotlib animation of the interpolated tissue force field.
    """
    initial_data = load_state(state_files[0])

    x_min, x_max, y_min, y_max = get_plot_limits(state_files)

    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))

    cell_points = initial_data["cell_points"]
    circumcenters = initial_data["circumcenters"]
    cell_triangles = initial_data["cell_triangles"]
    cell_types = initial_data["cell_types"]
    total_force = initial_data["total_force"]

    polygons, colours = get_real_polygons(cell_types, circumcenters, cell_triangles)

    real_mask = get_real_mask(cell_types)
    real_points = cell_points[real_mask]
    real_forces = total_force[real_mask]

    grid_fx = griddata(real_points, real_forces[:, 0], (grid_x, grid_y), method="linear")
    grid_fy = griddata(real_points, real_forces[:, 1], (grid_x, grid_y), method="linear")

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    polygon_collection = PolyCollection(polygons, facecolors=colours, edgecolors="black", linewidths=0.4, alpha=0.25, zorder=1)
    ax.add_collection(polygon_collection)

    boundary_points = cell_points[cell_types == 1]
    boundary_scatter = ax.scatter(boundary_points[:, 0], boundary_points[:, 1], color="green", s=8, alpha=0.4, zorder=2)

    quiver = ax.quiver(grid_x, grid_y, grid_fx, grid_fy, angles="xy", zorder=3)

    reporter = add_reporter(ax, initial_data)

    def update(frame):
        data = load_state(state_files[frame])

        cell_points = data["cell_points"]
        circumcenters = data["circumcenters"]
        cell_triangles = data["cell_triangles"]
        cell_types = data["cell_types"]
        total_force = data["total_force"]

        polygons, colours = get_real_polygons(cell_types, circumcenters, cell_triangles)
        polygon_collection.set_verts(polygons)
        polygon_collection.set_facecolors(colours)

        boundary_points = cell_points[cell_types == 1]
        boundary_scatter.set_offsets(boundary_points)

        real_mask = get_real_mask(cell_types)
        real_points = cell_points[real_mask]
        real_forces = total_force[real_mask]

        grid_fx = griddata(real_points, real_forces[:, 0], (grid_x, grid_y), method="linear")
        grid_fy = griddata(real_points, real_forces[:, 1], (grid_x, grid_y), method="linear")

        quiver.set_UVC(grid_fx, grid_fy)

        reporter.set_text(get_reporter_text(data))

        return polygon_collection, boundary_scatter, quiver, reporter

    return FuncAnimation(fig, update, frames=len(state_files), interval=100, blit=True)


def area_strain_animation(state_files, params):
    """
    Animate Voronoi cells coloured by relative area strain.

    Args:
        state_files: Ordered collection of saved simulation states.
        params: Static simulation parameters.

    Returns:
        Matplotlib animation of cell area strain.
    """
    initial_data = load_state(state_files[0])

    cell_points = initial_data["cell_points"]
    circumcenters = initial_data["circumcenters"]
    cell_triangles = initial_data["cell_triangles"]
    cell_types = initial_data["cell_types"]
    target_areas = initial_data["target_areas"]

    x_min, x_max, y_min, y_max = get_plot_limits(state_files)

    polygons, strains = get_area_strains(cell_types, target_areas, circumcenters, cell_triangles)

    min_strain = np.min(strains)
    max_strain = np.max(strains)

    if min_strain == max_strain:
        min_strain -= 1e-12
        max_strain += 1e-12

    norm = plt.Normalize(min_strain, max_strain)

    multiciliated_points = cell_points[cell_types == 2]
    boundary_points = cell_points[cell_types == 1]

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    polygon_collection = PolyCollection(polygons, array=strains, cmap="coolwarm", norm=norm, edgecolors="black", linewidths=0.5)
    ax.add_collection(polygon_collection)

    boundary_scatter = ax.scatter(boundary_points[:, 0], boundary_points[:, 1], color="green", s=10, zorder=2)
    multiciliated_scatter = ax.scatter(multiciliated_points[:, 0], multiciliated_points[:, 1], color="black", s=10, zorder=3)

    colorbar = fig.colorbar(polygon_collection, ax=ax)
    colorbar.set_label("Relative area strain")

    reporter = add_reporter(ax, initial_data)

    def update(frame):
        data = load_state(state_files[frame])

        cell_points = data["cell_points"]
        circumcenters = data["circumcenters"]
        cell_triangles = data["cell_triangles"]
        cell_types = data["cell_types"]
        target_areas = data["target_areas"]

        polygons, strains = get_area_strains(cell_types, target_areas, circumcenters, cell_triangles)

        min_strain = np.min(strains)
        max_strain = np.max(strains)

        if min_strain == max_strain:
            min_strain -= 1e-12
            max_strain += 1e-12

        polygon_collection.set_verts(polygons)
        polygon_collection.set_array(strains)
        polygon_collection.set_clim(min_strain, max_strain)

        boundary_points = cell_points[cell_types == 1]
        boundary_scatter.set_offsets(boundary_points)

        multiciliated_points = cell_points[cell_types == 2]
        multiciliated_scatter.set_offsets(multiciliated_points)

        colorbar.update_normal(polygon_collection)
        reporter.set_text(get_reporter_text(data))

        return polygon_collection, boundary_scatter, multiciliated_scatter, reporter

    return FuncAnimation(fig, update, frames=len(state_files), interval=100, blit=True)
