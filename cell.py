import numpy as np

class Cell():
    def __init__(self, id: int, x: float, y: float, neighbours: np.ndarray, area: float):
        self.id = id
        self.x = x
        self.y = y
        self.neighbours = neighbours
        self.area = area

    def calculate_attraction(self, cells: list):
        total_attraction = np.array([0.0, 0.0])
        position = np.array([self.x, self.y])
        for neighbour in self.neighbours:
            neighbour_position = np.array([cells[neighbour].x, cells[neighbour].y])
            difference = neighbour_position - position
            distance = np.linalg.norm(difference)
            unit_direction_vector = difference / distance
           
            # The simplest possible spring set up
            force = distance - 1 #+ (np.random.rand(1) * 0.0025)

            total_attraction += force * unit_direction_vector

        return total_attraction

    def calculate_repulsion(self, cells: list):
        total_repulsion = np.array([0, 0])

        return total_repulsion

class BasicCell(Cell):
    def step(self, cells: list):
        self.x, self.y = [self.x, self.y] + 0.95 * 0.0025 * (self.calculate_attraction(cells) - self.calculate_repulsion(cells))

class MulticiliatedCell(Cell):
    def step(self, cells: list):
        self.x, self.y = [self.x, self.y] + 0.95 * 0.0025 * (self.calculate_attraction(cells) - self.calculate_repulsion(cells))

class BorderCell(Cell):
    def step(self, cells: list):
        pass
