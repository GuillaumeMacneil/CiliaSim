import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PolyCollection
from scipy.interpolate import griddata


def load_state(state_file):
    with np.load(state_file) as data:
        return {key: data[key] for key in data.files}


def get_real_mask(cell_types):
    return (cell_types != 1) & (cell_types != -1)


def get_active_mask(cell_types):
    return cell_types != -1


def get_plot_limits(state_files, padding=1.0):
    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf

    for state_file in state_files:
        with np.load(state_file) as data:
            cell_points = data["cell_points"]
            cell_types = data["cell_types"]
            active_points = cell_points[get_active_mask(cell_types)]

            x_min = min(x_min, np.min(active_points[:, 0]))
            x_max = max(x_max, np.max(active_points[:, 0]))
            y_min = min(y_min, np.min(active_points[:, 1]))
            y_max = max(y_max, np.max(active_points[:, 1]))

    return x_min - padding, x_max + padding, y_min - padding, y_max + padding


def get_real_polygons(cell_types, circumcenters, cell_triangles):
    real_indices = np.where(get_real_mask(cell_types))[0]

    polygons = []
    colours = []

    for cell in real_indices:
        triangle_indices = cell_triangles[cell]
        triangle_indices = triangle_indices[triangle_indices >= 0]

        if len(triangle_indices) < 3:
            continue

        polygons.append(circumcenters[triangle_indices])
        colours.append("orange" if cell_types[cell] == 2 else "lightgray")

    return polygons, colours


def get_spring_edges(adjacency):
    rows, slots = np.where(adjacency >= 0)
    cols = adjacency[rows, slots]
    keep = rows < cols

    return np.column_stack((rows[keep], cols[keep]))


def get_spring_segments(cell_points, adjacency):
    edges = get_spring_edges(adjacency)

    if len(edges) == 0:
        return np.empty((0, 2, 2)), edges

    segments = np.stack((cell_points[edges[:, 0]], cell_points[edges[:, 1]]), axis=1)

    return segments, edges


def get_major_axis_segments(polygons):
    segments = []

    for polygon in polygons:
        coords = compute_major_axis(polygon)

        if coords is None:
            continue

        x1, y1, x2, y2 = coords
        segments.append(((x1, y1), (x2, y2)))

    return np.asarray(segments)


def get_spring_forces(cell_points, edges, spring_length, spring_constant):
    if len(edges) == 0:
        return np.empty(0)

    differences = cell_points[edges[:, 1]] - cell_points[edges[:, 0]]
    distances = np.linalg.norm(differences, axis=1)

    return spring_constant * (distances - spring_length)


def get_area_strains(cell_types, target_areas, circumcenters, cell_triangles):
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

        area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))

        polygons.append(polygon)
        strains.append((area - target_areas[cell]) / target_areas[cell])

    return polygons, np.asarray(strains)


def basic_animation(state_files, params):
    initial_data = load_state(state_files[0])

    cell_points = initial_data["cell_points"]
    circumcenters = initial_data["circumcenters"]
    cell_triangles = initial_data["triangles"]
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

        return polygon_collection, boundary_scatter

    return FuncAnimation(fig, update, frames=len(state_files), interval=100, blit=True)


def spring_animation(state_files, params):
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
            frame_segments, frame_edges = get_spring_segments(data["cell_points"], data["adjacency"])
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

        return polygon_collection, boundary_scatter, spring_collection

    return FuncAnimation(fig, update, frames=len(state_files), interval=100, blit=True)


def compute_major_axis(polygon):
    pts = np.asarray(polygon, dtype=np.float64)
    if pts.size == 0 or pts.shape[0] < 2:
        return None

    center = pts.mean(axis=0)
    diffs = pts - center
    if np.allclose(diffs, 0):
        return None

    # PCA: largest eigenvector
    cov = (diffs.T @ diffs) / float(max(1, len(pts)))
    try:
        _, eigvecs = np.linalg.eigh(cov)
    except Exception:
        return None
    v = eigvecs[:, -1]
    vnorm = np.linalg.norm(v)
    if vnorm == 0:
        return None
    v = v / vnorm

    # helper: intersection of infinite line (center + t*v) with segment (a -> b)
    def line_segment_intersection(p0, v, a, b, tol=1e-12):
        # Solve: a + s*(b-a) = p0 + t*v  ->  s*(b-a) - t*v = p0 - a
        M = np.column_stack((b - a, -v))  # 2x2
        det = np.linalg.det(M)
        if abs(det) < tol:
            return None
        rhs = p0 - a
        try:
            sol = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            return None
        s = sol[0]
        # s in [0,1] => intersection falls on the segment
        if s < -1e-9 or s > 1 + 1e-9:
            return None
        t = sol[1]
        return p0 + t * v

    # collect intersections with all polygon edges
    intersections = []
    N = pts.shape[0]
    for i in range(N):
        a = pts[i]
        b = pts[(i + 1) % N]
        ip = line_segment_intersection(center, v, a, b)
        if ip is not None:
            intersections.append(ip)

    # unique intersections (within tolerance)
    if len(intersections) > 0:
        ints = np.vstack(intersections)
        # remove duplicates
        uniq = []
        for p in ints:
            if not any(np.allclose(p, q, atol=1e-8) for q in uniq):
                uniq.append(p)
        ints = np.vstack(uniq)

    # If we have at least two intersections, pick min & max along v
    if 'ints' in locals() and ints.shape[0] >= 2:
        ts = (ints - center) @ v
        tmin = ts.min()
        tmax = ts.max()
        p1 = center + v * tmin
        p2 = center + v * tmax
        return (p1[0], p1[1], p2[0], p2[1])

    # Fallback: use extreme projected vertices (guaranteed on boundary)
    proj_t = (pts - center) @ v
    tmin = proj_t.min()
    tmax = proj_t.max()
    p1 = center + v * tmin
    p2 = center + v * tmax
    return (p1[0], p1[1], p2[0], p2[1])


def major_axes_animation(state_files, params):
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

        return polygon_collection, axis_collection, boundary_scatter

    return FuncAnimation(fig, update, frames=len(state_files), interval=100, blit=True)


def boundary_animation(state_files, params):
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

        return polygon_collection, boundary_scatter, boundary_line

    return FuncAnimation(fig, update, frames=len(state_files), interval=100, blit=True)


def force_vectors_animation(state_files, params):
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

        return polygon_collection, boundary_scatter, quiver

    return FuncAnimation(fig, update, frames=len(state_files), interval=100, blit=True)


def force_field_animation(state_files, params, grid_size=50):
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

    # Faint tissue underneath
    polygon_collection = PolyCollection(polygons, facecolors=colours, edgecolors="black", linewidths=0.4, alpha=0.25, zorder=1)
    ax.add_collection(polygon_collection)

    # Faint boundary points
    boundary_points = cell_points[cell_types == 1]
    boundary_scatter = ax.scatter(boundary_points[:, 0], boundary_points[:, 1], color="green", s=8, alpha=0.4, zorder=2)

    # Force field on top
    quiver = ax.quiver(grid_x, grid_y, grid_fx, grid_fy, angles="xy", zorder=3)

    def update(frame):
        data = load_state(state_files[frame])

        cell_points = data["cell_points"]
        circumcenters = data["circumcenters"]
        cell_triangles = data["cell_triangles"]
        cell_types = data["cell_types"]
        total_force = data["total_force"]

        # Update tissue
        polygons, colours = get_real_polygons(cell_types, circumcenters, cell_triangles)
        polygon_collection.set_verts(polygons)
        polygon_collection.set_facecolors(colours)

        # Update boundary points
        boundary_points = cell_points[cell_types == 1]
        boundary_scatter.set_offsets(boundary_points)

        # Update force field
        real_mask = get_real_mask(cell_types)
        real_points = cell_points[real_mask]
        real_forces = total_force[real_mask]

        grid_fx = griddata(real_points, real_forces[:, 0], (grid_x, grid_y), method="linear")
        grid_fy = griddata(real_points, real_forces[:, 1], (grid_x, grid_y), method="linear")

        quiver.set_UVC(grid_fx, grid_fy)

        return polygon_collection, boundary_scatter, quiver

    return FuncAnimation(fig, update, frames=len(state_files), interval=100, blit=True)


def area_strain_animation(state_files, params):
    initial_data = load_state(state_files[0])

    cell_points = initial_data["cell_points"]
    circumcenters = initial_data["circumcenters"]
    cell_triangles = initial_data["cell_triangles"]
    cell_types = initial_data["cell_types"]
    target_areas = initial_data["target_areas"]

    x_min, x_max, y_min, y_max = get_plot_limits(state_files)

    # Determine a single colour scale for the entire animation
    max_strain = 1e-12

    for state_file in state_files:
        with np.load(state_file) as data:
            _, strains = get_area_strains(data["cell_types"], data["target_areas"], data["circumcenters"], data["cell_triangles"])

            if len(strains) > 0:
                max_strain = max(max_strain, np.max(np.abs(strains)))

    norm = plt.Normalize(-max_strain, max_strain)
    cmap = plt.get_cmap("coolwarm")

    polygons, strains = get_area_strains(cell_types, target_areas, circumcenters, cell_triangles)

    multiciliated_points = cell_points[cell_types == 2]
    boundary_points = cell_points[cell_types == 1]

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    polygon_collection = PolyCollection(polygons, array=strains, cmap=cmap, norm=norm, edgecolors="black", linewidths=0.5)
    ax.add_collection(polygon_collection)

    boundary_scatter = ax.scatter(boundary_points[:, 0], boundary_points[:, 1], color="green", s=10, zorder=2)

    multiciliated_scatter = ax.scatter(multiciliated_points[:, 0], multiciliated_points[:, 1], color="black", s=10, zorder=3)

    colorbar = fig.colorbar(polygon_collection, ax=ax)
    colorbar.set_label("Relative area strain")

    def update(frame):
        data = load_state(state_files[frame])

        cell_points = data["cell_points"]
        circumcenters = data["circumcenters"]
        cell_triangles = data["cell_triangles"]
        cell_types = data["cell_types"]
        target_areas = data["target_areas"]

        polygons, strains = get_area_strains(cell_types, target_areas, circumcenters, cell_triangles)

        polygon_collection.set_verts(polygons)
        polygon_collection.set_array(strains)

        boundary_points = cell_points[cell_types == 1]
        boundary_scatter.set_offsets(boundary_points)

        multiciliated_points = cell_points[cell_types == 2]
        multiciliated_scatter.set_offsets(multiciliated_points)

        return polygon_collection, boundary_scatter, multiciliated_scatter

    return FuncAnimation(fig, update, frames=len(state_files), interval=100, blit=True)
