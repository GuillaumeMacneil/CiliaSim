# CiliaSim

CiliaSim is a two-dimensional aperiodic mechanical tissue simulation for studying the behaviour of multiciliated cells (MCCs) and exogenous fluid flows within an epithelial tissue.

Cells are represented by Voronoi regions generated from cell-centre positions. The corresponding Delaunay triangulation defines cell connectivity. Tissue mechanics are described by an energy model incorporating preferred neighbour spacing, cell area, and cell perimeter. Forces are obtained automatically from the gradient of this energy using JAX.

The model supports externally applied ciliary and flow forces, adaptive ghost-cell boundaries, multiple tissue initialization strategies, simulation state saving, and several animations for examining tissue mechanics.

## Installation

CiliaSim uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

Clone the repository and install the environment:

```bash
git clone <repository-url>
cd CiliaSim
uv sync
```

Run scripts using:

```bash
uv run python scripts/tissue_test.py
```

For GPU-accelerated JAX, a compatible CUDA-enabled JAX installation and NVIDIA driver are required.

## Basic usage

A tissue can be initialized and simulated with:

```python
import numpy as np

from ciliasim import tissue


simulation = tissue.Tissue(
    x=30,
    y=20,
    density=0.06,
    spring_length=1.0,
    spring_constant=1.0,
    area_constant=1.0,
    perimeter_constant=1.0,
    random_layout=True,
    save=True,
    save_freq=10,
    output_dir="simulations/",
)

simulation.specify_cells((np.sqrt(3) / 2, 0.0))

simulation.simulate(1000)

simulation.set_uniform_ciliary_forces(
    np.array([0.0, 1.0]),
    magnitude=1.0,
)

simulation.simulate(2000)
```

For a regular triangular lattice with nearest-neighbour spacing $l_0$, the corresponding regular hexagonal Voronoi area is:

```math
A_0 = \frac{\sqrt{3}}{2}l_0^2
```

### Using the templates

The `templates/` directory contains simple starting points for common simulation configurations:

| Template | Description |
| --- | --- |
| `hexagonal_center.py` | Hexagonal tissue with a single central MCC |
| `random_center.py` | Random tissue with a single central MCC |
| `random_tissue.py` | Random tissue with MCCs distributed throughout the tissue |
| `examine.py` | Load and visualize a saved simulation |

The simulation templates can be run directly with `uv`. For example:

```bash
uv run templates/hexagonal_center.py
```

Each template exposes the main simulation parameters near the top of the file and is intended to be copied or edited for a particular experiment.

For example, change:

```python
N = 20
```

to alter the requested tissue dimensions, or modify the simulation and force parameters passed to `Tissue`.

To visualize a saved simulation, run:

```bash
uv run templates/examine.py
```

By default, `examine.py` loads the most recent simulation in `simulations/`, using the timestamp encoded in the simulation directory name and falling back to its modification time if necessary.

To examine a particular simulation instead, set `simulation_dir` directly in the template:

```python
simulation_dir = Path(
    "simulations/25-11-26_14-23-26_20x20_center_hexagonal/"
)
```

The visualization type can be changed through the `plot_type` argument:

```python
simulation.animate(
    plot_type="major-axes",
    save=True,
)
```

Available plot types include `basic`, `spring`, `major-axes`, `boundary`, `force-vectors`, `force-field`, and `area-strain`.

## Tissue initialization

CiliaSim currently supports two initial tissue layouts.

### Hexagonal layout

The hexagonal layout constructs a complete triangular lattice using axial coordinates. The number of cells is reduced to the largest complete set of concentric hexagonal rings below the requested number of cells.

For this layout, the ghost boundary is known exactly and is constructed as the next axial lattice ring.

### Random layout

Random tissues are initialized using Poisson-disk sampling.

A ghost-cell boundary is constructed outside the convex hull of the real tissue. During simulation, boundary cells can be added or removed as the tissue changes shape.

## Cell types

The simulation uses integer cell types:

| Value | Cell type |
| ---: | --- |
| `-1` | Vacant storage slot |
| `0` | Non-multiciliated tissue cell |
| `1` | Ghost boundary cell |
| `2` | Multiciliated cell |

## Geometry and connectivity

Cell centres define a Delaunay triangulation. The dual Voronoi tessellation represents the physical cell regions.

For each cell, CiliaSim stores:

- neighbouring cell indices;
- incident Delaunay triangles;
- an ordered list of incident triangles;
- the resulting Voronoi polygon;
- cell area and perimeter.

Voronoi vertices are obtained from the circumcentres of incident Delaunay triangles.

Boundary cells are constrained to an ordered cycle to prevent unwanted Delaunay connections between non-consecutive ghost cells.

## Mechanical model

The current tissue energy contains three contributions:

```math
E =
E_{\mathrm{spring}}
+
E_{\mathrm{area}}
+
E_{\mathrm{perimeter}}
```

### Spring energy

Neighbouring cell centres are connected by springs with preferred length $l_0$:

```math
E_{\mathrm{spring}}
=
\frac{k_s}{2}
\sum_{\langle i,j\rangle}
\left(d_{ij}-l_0\right)^2
```

where $d_{ij}$ is the distance between neighbouring cell centres $i$ and $j$.

This penalizes deviations from the preferred centre-to-centre separation.

### Area energy

Real cells have a preferred area $A_i^0$:

```math
E_{\mathrm{area}}
=
\frac{k_A}{2}
\sum_i
\left(A_i-A_i^0\right)^2
```

Ghost boundary cells do not contribute to the area energy.

### Perimeter energy

Cells additionally have a preferred perimeter:

```math
E_{\mathrm{perimeter}}
=
\frac{k_P}{2}
\sum_i
\left(P_i-P_0\right)^2
```

The current preferred perimeter corresponds to a regular hexagonal Voronoi cell at the preferred spring spacing:

```math
P_0 = 2\sqrt{3}\,l_0
```

For $l_0=1$, the compatible regular-lattice geometry is therefore:

```math
A_0 = \frac{\sqrt{3}}{2},
\qquad
P_0 = 2\sqrt{3}
```

Thus the spring, area, and perimeter terms share the same preferred geometry for a regular hexagonal tissue.

## Forces and dynamics

Internal mechanical forces are calculated from the negative gradient of the mechanical energy with respect to cell-centre position:

```math
\mathbf{F}_i^{\mathrm{internal}}
=
-\nabla_{\mathbf{r}_i}E
```

The gradient is calculated automatically using JAX and JIT-compiled for repeated simulation steps.

Ciliary forces can be assigned uniformly:

```python
simulation.set_uniform_ciliary_forces(
    np.array([0.0, 1.0]),
    magnitude=0.5,
)
```

or with random orientations:

```python
simulation.set_random_ciliary_forces(
    magnitude=0.5,
)
```

A global flow force can also be applied to MCCs:

```python
simulation.set_flow_force(
    np.array([1.0, 0.0]),
    magnitude=0.1,
)
```

The total force acting on a cell is:

```math
\mathbf{F}_i
=
\mathbf{F}_i^{\mathrm{internal}}
+
\mathbf{F}_i^{\mathrm{cilia}}
+
\mathbf{F}_i^{\mathrm{flow}}
```

The flow contribution is applied only to multiciliated cells.

The simulation uses overdamped dynamics:

```math
\Delta\mathbf{r}_i
=
\mu\,\mathbf{F}_i\,\Delta t
```

where $\mu$ is the cell mobility and $\Delta t$ is the simulation timestep.

## Boundary handling

Ghost cells surround the real tissue so that peripheral cells retain closed Voronoi regions.

The boundary is represented by:

- `boundary_cycle`, an ordered sequence of ghost-cell indices;
- `boundary_cycle_mask`, a Boolean mask identifying boundary cells.

During simulation, the boundary is evaluated after cell movement. Ghost cells may be added or removed where necessary, after which active cell arrays are compacted and the Delaunay connectivity is recomputed.

Boundary connectivity is constrained so that a ghost cell can connect to its two adjacent boundary cells and to interior cells, but not arbitrarily to other boundary cells.

## Saving simulations

When saving is enabled, CiliaSim creates a timestamped simulation directory containing:

```text
simulation/
├── params.json
├── 000000.npz
├── 000010.npz
├── 000020.npz
└── ...
```

`params.json` contains static simulation parameters.

Each NPZ state contains simulation data including:

- iteration;
- cell positions;
- cell types;
- target areas;
- Delaunay triangles;
- cell-triangle connectivity;
- adjacency;
- circumcentres;
- boundary cycle and mask;
- internal forces;
- ciliary forces;
- flow force;
- total force.

## Visualization

Saved simulations can be examined using `Examine`:

```python
from ciliasim.examine import Examine


simulation = Examine(
    "simulations/<simulation-directory>",
    output_dir="animations/",
)

simulation.animate(
    plot_type="basic",
    show=True,
    save=False,
)
```

Available animation types include:

| Plot type | Description |
| --- | --- |
| `basic` | Voronoi tissue geometry |
| `spring` | Springs coloured by compression and extension |
| `major-axes` | Major axes of Voronoi cell polygons |
| `boundary` | Explicit ghost-cell boundary cycle |
| `force-vectors` | Cell force vectors |
| `force-field` | Interpolated tissue force field |
| `area-strain` | Relative cell area strain |

Animations also display a reporter containing the current iteration, cell counts, and force statistics.

Animations can be saved as MP4 files using FFmpeg.

## Project structure

The core package is organized as:

```text
src/ciliasim/
├── boundary.py
├── connectivity.py
├── examine.py
├── geometry.py
├── initialisation.py
├── plotting.py
└── tissue.py
```

### `tissue.py`

Defines the tissue state, mechanical energy, force calculation, simulation loop, MCC specification, and state saving.

### `geometry.py`

Contains differentiable JAX functions for calculating triangle circumcentres and Voronoi cell areas and perimeters.

### `connectivity.py`

Constructs Delaunay triangulations, adjacency arrays, and ordered cell-triangle connectivity.

### `boundary.py`

Creates and maintains ghost-cell boundaries and constrains their connectivity.

### `initialisation.py`

Provides random and hexagonal tissue layouts and target-area initialization.

### `plotting.py`

Contains geometry, force, strain, spring, boundary, and cell-shape animations.

### `examine.py`

Provides an interface for loading, displaying, and exporting saved simulations.

## Current limitations

The model uses an ordinary Voronoi tessellation in which cell boundaries are determined entirely by cell-centre positions. Consequently, strongly forced cells can exhibit unintuitive elongation because translating a generator also moves its Voronoi interfaces.

The current implementation also performs complete Delaunay retriangulation when updating tissue connectivity. Partial or local retriangulation is not yet implemented.

## Dependencies

Major dependencies include:

- NumPy
- SciPy
- JAX
- Matplotlib
- tqdm

Exact dependency versions are recorded in `uv.lock`.

## Development

Install the locked environment with:

```bash
uv sync
```

Run commands within the environment using:

```bash
uv run <command>
```

The `uv.lock` file should be committed to version control to keep development and simulation environments reproducible.
