# CiliaSim
A Voronoi-based aperiodic tissue model for simulating multiciliated tissues under exogenous flow forces.

> **⚠️ Please note that this project is under active development - all functionality is provided as-is and breaking changes may still occur. ⚠️**

## Installation (Unix-based systems)
This project is written in `Python 3` and requires the  `numpy`, `scipy`, `matplotlib`, `tqdm`, `numba` libraries. Assuming `Python 3` has already been downloaded, follow the steps below:

```bash
$ git clone https://github.com/GuillaumeMacneil/CiliaSim
$ cd CiliaSim
$ python3 -m venv venv
$ pip install .
```
These commands set up a virtual environment (venv) suitable for running the CiliaSim software. With each new terminal session, the command `source venv/bin/activate` will have to be run to enter the venv.

## Basic Use
Essentially, CiliaSim provides a framework for defining a 2D ciliated tissue, describing the forces it will experience, and the timescale on this process will occur. We may define a basic simulation like so:
```python
from tissue import Tissue

tissue = Tissue(15, 15, 0.06)
tissue.hexagonal_grid_layout()
tissue.set_plot_spring()
tissue.simulate("Tissue annealing - No forces.", 1000, 100)
tissue.set_uniform_cilia_forces([0, 1], 0.5)
tissue.simulate("Tissue under upward cilia force of magnitude 0.5.", 9000, 100)
``` 

This simple definition will generate a 15x15 tissue with a 6% cilia density in a hexagonal grid layout. The ciliated cells will be randomly distributed over the tissue and, when plotted, the tissue will show the springs connecting cell centers coloured by strain. The first 1000 iterations of the simulation is spent "annealing" the tissue, and then a 0.5 magnitude upward force is applied to each ciliated cell and the simulation is continued for a further 9000 iterations.

## Project Structure
The project consists of the following files:
* **`tissue.py`** -> Describes the `Tissue` class, its dynamics and any other supporting functions (logging, plot selection, etc.)
* **`plotting.py`** -> Contains all of the plotting functions.
* **`manager.py`** -> Loads saved tissue `.json` files, allowing for arbitrary plotting and navigation forward and backward in time.
* **`jit_functions.py`** -> Contains the most compute-heavy functions, with `numba` decorators for JIT compilation.

## Important Notes
As stated above, this project is still under active development. It is likely that the structure and use of CiliaSim will change significantly in the next few months. The following features will be implemented in the near future:
1. Replacing the `write_to_file(...)` functionality and ".json" format to be more easily interpreted and faster to load.
2. Updating the (currently broken) `Manager` class to accomodate the recent changes.
3. Changing the way plotting is performed to allow for the generation of animations which can be played at any speed.
4. Re-implementing the energy calculations to allow tracking of the net energy in a tissue over time.
5. Making the appropriate structual changes to make CiliaSim an import-able Python 3 library.
6. Link to the paper you based this off.


# Thoughts

- [ ] Preallocate an array of a certain size.
- [ ] Only voronoi once the positions have shifted more than some tolerance (this is a nlgn operation otherwise)
- [ ] generalise to nd.