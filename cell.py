import numpy as np

class Cell():
    def __init__(self, id: int, x: float, y: float, neighbours: np.ndarray, area: float):
        self.id = id
        self.x = x
        self.y = y
        self.neighbours = neighbours
        self.area = area

class BasicCell(Cell):
    pass

class MulticiliatedCell(Cell):
    pass

class BorderCell(Cell):
    pass
