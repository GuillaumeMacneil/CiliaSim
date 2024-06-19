import numpy as np

class Cell():
    def __init__(self, id: int, x: float, y: float, neighbours: np.ndarray):
        self.id = id
        self.x = x
        self.y = y
        self.neighbours = neighbours

class BasicCell(Cell):
    pass

class MulticiliatedCell(Cell):
    pass
