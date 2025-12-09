import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection

def get_voronoi_polygons(cell_points, cell_types, circumcenters, triangles, num_cells):
    polygons = []
    for cell_index in range(num_cells):
        if cell_types[cell_index] == 1:
            polygons.append([])
            continue
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
    polygons = get_voronoi_polygons(cell_points, cell_types, circumcenters, triangles, num_cells)

    # Set up plot configuration
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(np.min(cell_points[:, 0]) - 1, np.max(cell_points[:, 0]) + 1)
    ax.set_ylim(np.min(cell_points[:, 1]) - 1, np.max(cell_points[:, 1]) + 1)   
    
    # Draw cell polygon patches
    patches = [None] * num_cells
    for i in range(num_cells):
        polygon = polygons[i]
        if cell_types[i] == 1 or len(polygon) == 0:
            continue

        colour = "orange" if cell_types[i] == 2 else "lightgray"
        patch = Polygon(polygon, closed=True, facecolor=colour, edgecolor='black', linewidth=0.5)
        ax.add_patch(patch)
        patches[i] = patch

    # Draw boundary points
    boundary_mask = cell_types == 1
    boundary_points = cell_points[boundary_mask]
    boundary_scatter = ax.scatter(boundary_points[:, 0], boundary_points[:, 1], color="green", s=10)

    def update(frame):
        # Load the data required for a basic voronoi plot
        data = np.load(state_files[frame])
        cell_points = data["cell_points"]
        circumcenters = data["circumcenters"]
        triangles = data["triangles"]
        cell_types = data["cell_types"]
        num_cells = np.sum(cell_types != -1)

        # Clear old patches
        for p in patches:
            if p is not None:
                p.remove()
        patches.clear()

        # Rebuild new patches
        polygons = get_voronoi_polygons(cell_points, cell_types, circumcenters, triangles, num_cells)
        for i in range(num_cells):
            polygon = polygons[i]
            if cell_types[i] == 1 or len(polygon) == 0:
                patches.append(None)
                continue

            colour = "orange" if cell_types[i] == 2 else "lightgray"
            patch = Polygon(polygon, closed=True, facecolor=colour, edgecolor='black', linewidth=0.5)
            ax.add_patch(patch)
            patches.append(patch)

        # Update boundary points
        boundary_mask = cell_types == 1
        boundary_points = cell_points[boundary_mask]
        boundary_scatter.set_offsets(boundary_points)

        return patches + [boundary_scatter]
       
    return FuncAnimation(fig, update, frames=len(state_files), interval=100, blit=False)


def spring_animation(state_files, params):
    # Load the initial points and set up the plot
    initial_data = np.load(state_files[0])
    cell_points = initial_data["cell_points"]
    circumcenters = initial_data["circumcenters"]
    triangles = initial_data["triangles"]
    cell_types = initial_data["cell_types"]
    adjacency = initial_data["adjacency"]
    num_cells = np.sum(cell_types != -1)
    polygons = get_voronoi_polygons(cell_points, cell_types, circumcenters, triangles, num_cells)

    # Set up plot configuration
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(np.min(cell_points[:, 0]) - 1, np.max(cell_points[:, 0]) + 1)
    ax.set_ylim(np.min(cell_points[:, 1]) - 1, np.max(cell_points[:, 1]) + 1)   
    
    # Draw cell polygon patches
    patches = [None] * num_cells
    for i in range(num_cells):
        polygon = polygons[i]
        if cell_types[i] == 1 or len(polygon) == 0:
            continue

        colour = "orange" if cell_types[i] == 2 else "lightgray"
        patch = Polygon(polygon, closed=True, facecolor=colour, edgecolor='black', linewidth=0.5)
        ax.add_patch(patch)
        patches[i] = patch

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
        spring_pairs = []
        if b in adjacency[a]:
            spring_pairs.append([cell_points[a], cell_points[b]])
        if c in adjacency[b]:
            spring_pairs.append([cell_points[b], cell_points[c]])
        if a in adjacency[c]:
            spring_pairs.append([cell_points[c], cell_points[a]])

        springs.extend(spring_pairs)
    
    springs_line_collection = LineCollection(springs, colors="gray", linewidths=0.5)
    ax.add_collection(springs_line_collection)

    def update(frame):
        # Load the data required for a basic voronoi plot
        data = np.load(state_files[frame])
        cell_points = data["cell_points"]
        circumcenters = data["circumcenters"]
        triangles = data["triangles"]
        cell_types = data["cell_types"]
        adjacency = data["adjacency"]
        num_cells = np.sum(cell_types != -1)

        # Clear old patches
        for p in patches:
            if p is not None:
                p.remove()
        patches.clear()

        # Rebuild new patches
        polygons = get_voronoi_polygons(cell_points, cell_types, circumcenters, triangles, num_cells)
        for i in range(num_cells):
            polygon = polygons[i]
            if cell_types[i] == 1 or len(polygon) == 0:
                patches.append(None)
                continue

            colour = "orange" if cell_types[i] == 2 else "lightgray"
            patch = Polygon(polygon, closed=True, facecolor=colour, edgecolor='black', linewidth=0.5)
            ax.add_patch(patch)
            patches.append(patch)

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
            spring_pairs = []
            if b in adjacency[a]:
                spring_pairs.append([cell_points[a], cell_points[b]])
            if c in adjacency[b]:
                spring_pairs.append([cell_points[b], cell_points[c]])
            if a in adjacency[c]:
                spring_pairs.append([cell_points[c], cell_points[a]])

            springs.extend(spring_pairs)

        springs_line_collection.set_segments(springs)

        artists = [patch for patch in patches if patch is not None]
        return artists + [boundary_scatter, springs_line_collection]
       
    return FuncAnimation(fig, update, frames=len(state_files), interval=100, blit=True)

def compute_major_axis(polygon):
    """
    polygon: sequence of (x,y) coords (Nx2)
    returns: (x1,y1,x2,y2) endpoints lying on the polygon boundary (or None)
    """
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
        eigvals, eigvecs = np.linalg.eigh(cov)
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
    # Load the initial points and set up the plot
    initial_data = np.load(state_files[0])
    cell_points = initial_data["cell_points"]
    circumcenters = initial_data["circumcenters"]
    triangles = initial_data["triangles"]
    cell_types = initial_data["cell_types"]
    num_cells = np.sum(cell_types != -1)
    polygons = get_voronoi_polygons(cell_points, cell_types, circumcenters, triangles, num_cells)

    # Set up plot configuration
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(np.min(cell_points[:, 0]) - 1, np.max(cell_points[:, 0]) + 1)
    ax.set_ylim(np.min(cell_points[:, 1]) - 1, np.max(cell_points[:, 1]) + 1)

    # Draw cell polygon patches
    patches = [None] * num_cells
    for i in range(num_cells):
        polygon = polygons[i]
        if cell_types[i] == 1 or len(polygon) == 0:
            continue

        colour = "orange" if cell_types[i] == 2 else "lightgray"
        patch = Polygon(polygon, closed=True, facecolor=colour, edgecolor='black', linewidth=0.5)
        ax.add_patch(patch)
        patches[i] = patch

    # Draw boundary points
    boundary_mask = cell_types == 1
    boundary_points = cell_points[boundary_mask]
    boundary_scatter = ax.scatter(boundary_points[:, 0], boundary_points[:, 1], color="green", s=10)

    # Draw major-axis lines initially
    axis_lines = [None] * num_cells
    for i in range(num_cells):
        polygon = polygons[i]
        if cell_types[i] == 1 or len(polygon) == 0:
            continue
        coords = compute_major_axis(polygon)
        if coords is None:
            axis_lines[i] = None
            continue
        x1, y1, x2, y2 = coords
        ln, = ax.plot([x1, x2], [y1, y2], color="red", linewidth=1)
        axis_lines[i] = ln

    def update(frame):
        # Load the data required for a basic voronoi plot
        data = np.load(state_files[frame])
        cell_points = data["cell_points"]
        circumcenters = data["circumcenters"]
        triangles = data["triangles"]
        cell_types = data["cell_types"]
        num_cells = np.sum(cell_types != -1)

        # Clear old patches
        for p in patches:
            if p is not None:
                p.remove()
        patches.clear()

        # Also remove old axis lines
        for ln in axis_lines:
            if ln is not None:
                try:
                    ln.remove()
                except Exception:
                    pass
        axis_lines.clear()

        # Rebuild new patches
        polygons = get_voronoi_polygons(cell_points, cell_types, circumcenters, triangles, num_cells)
        for i in range(num_cells):
            polygon = polygons[i]
            if cell_types[i] == 1 or len(polygon) == 0:
                patches.append(None)
                continue

            colour = "orange" if cell_types[i] == 2 else "lightgray"
            patch = Polygon(polygon, closed=True, facecolor=colour, edgecolor='black', linewidth=0.5)
            ax.add_patch(patch)
            patches.append(patch)

        # Recreate major-axis lines for this frame
        for i in range(num_cells):
            polygon = polygons[i]
            if cell_types[i] == 1 or len(polygon) == 0:
                axis_lines.append(None)
                continue
            coords = compute_major_axis(polygon)
            if coords is None:
                axis_lines.append(None)
                continue
            x1, y1, x2, y2 = coords
            ln, = ax.plot([x1, x2], [y1, y2], color="red", linewidth=1)
            axis_lines.append(ln)

        # Update boundary points
        boundary_mask = cell_types == 1
        boundary_points = cell_points[boundary_mask]
        if boundary_points.size > 0:
            boundary_scatter.set_offsets(boundary_points)
        else:
            boundary_scatter.set_offsets(np.empty((0, 2)))

        return patches + axis_lines + [boundary_scatter]

    return FuncAnimation(fig, update, frames=len(state_files), interval=100, blit=False)
