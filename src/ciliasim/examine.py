import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from ciliasim import plotting

class Examine():
    def __init__(self, simulation_dir: str, output_dir: str = ""):
        # Set up the paths to read from and output to
        self.simulation_dir = Path(simulation_dir)
        if output_dir:
            self.output_path = Path(output_dir) / self.simulation_dir.name 
            self.output_path.mkdir(parents=True, exist_ok=True)

        # Load the static parameters
        with open(self.simulation_dir / "params.json", "r") as params_file:
            self.params = json.load(params_file)

    def animate(self, plot_type: str = "basic", show: bool = True, save: bool = False):
        # Choose a plot type and animate the simulation
        animate = plotting.basic_animation
        match plot_type:
            case "spring":
                animate = plotting.spring_animation
            case "major-axes":
                animate = plotting.major_axes_animation
            case _:
                pass
        
        state_files = sorted(self.simulation_dir.glob("*.npz"))
        animation = animate(state_files, self.params)
         
        if show:
            plt.show()
        
        # Save the animation as a .mp4
        if save:
            if not self.output_path:
                raise ValueError("Both `save` and `output_dir` must be provided for animation saving.")
            
            timestamp = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
            animation.save(self.output_path / f"{timestamp}_{plot_type}.mp4", fps=5, writer="ffmpeg")
