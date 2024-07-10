import numpy as np
from functions import *

class Cell():
    def __init__(self, id: int, x: float, y: float, neighbours: np.ndarray):
        self.id = id
        self.x = x
        self.y = y
        self.neighbours = neighbours
        self.area = None

    def set_area(self, area: float):
        self.area = area

    def step(self, force_matrix: np.ndarray, annealing: bool = False):
        self.x, self.y = [self.x, self.y] + 0.95 * 0.01 * np.sum(force_matrix[self.id, self.neighbours], axis=0)

class BasicCell(Cell):
    pass

class MulticiliatedCell(Cell):
    pass

class BorderCell(Cell):
    pass
