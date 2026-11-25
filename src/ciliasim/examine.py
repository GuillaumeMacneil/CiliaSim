import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from ciliasim import plotting


class Examine:
    """Load and visualize the saved states of a simulation."""

    def __init__(self, simulation_dir: str, output_dir: str = ""):
        """
        Initialize an interface for examining a saved simulation.

        Args:
            simulation_dir: Directory containing simulation states and parameters.
            output_dir: Optional directory in which generated outputs are saved.

        Returns:
            None.
        """
        self.simulation_dir = Path(simulation_dir)
        self.output_path = None
        if output_dir:
            self.output_path = Path(output_dir) / self.simulation_dir.name
            self.output_path.mkdir(parents=True, exist_ok=True)

        with open(self.simulation_dir / "params.json", "r") as params_file:
            self.params = json.load(params_file)

    def animate(self, plot_type: str = "basic", show: bool = True, save: bool = False):
        """
        Create an animation from the saved simulation states.

        Args:
            plot_type: Type of simulation visualization to generate.
            show: Whether to display the animation interactively.
            save: Whether to save the animation as an MP4 file.

        Returns:
            None.
        """
        # Choose a plot type and animate the simulation
        animate = plotting.basic_animation
        match plot_type:
            case "spring":
                animate = plotting.spring_animation
            case "major-axes":
                animate = plotting.major_axes_animation
            case "boundary":
                animate = plotting.boundary_animation
            case "force-vectors":
                animate = plotting.force_vectors_animation
            case "force-field":
                animate = plotting.force_field_animation
            case "area-strain":
                animate = plotting.area_strain_animation
            case _:
                pass

        state_files = sorted(self.simulation_dir.glob("*.npz"))
        animation = animate(state_files, self.params)

        if show:
            plt.show()

        # Save the animation as a .mp4
        if save:
            if not self.output_path:
                raise ValueError(
                    "Both `save` and `output_dir` must be provided for animation saving."
                )

            timestamp = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
            animation.save(
                self.output_path / f"{timestamp}_{plot_type}.mp4",
                fps=5,
                writer="ffmpeg",
            )
