import numpy as np
from functions import *

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
            force = distance
            
            total_attraction += force * unit_direction_vector

        return total_attraction

    def calculate_repulsion(self, cells: list):
        total_repulsion = np.array([0.0, 0.0])
        position = np.array([self.x, self.y])
        neighbour_positions = np.array([np.array([cells[neighbour].x, cells[neighbour].y]) for neighbour in self.neighbours])
        neighbourhood_area = polygon_area(neighbour_positions) 
        area_difference = neighbourhood_area - self.area 

        for neighbour in self.neighbours:
            neighbour_position = np.array([cells[neighbour].x, cells[neighbour].y])
            difference = neighbour_position - position
            distance = np.linalg.norm(difference)
            unit_direction_vector = difference / distance

            force = area_difference

            total_repulsion += force * unit_direction_vector

        return total_repulsion

    def set_area(self, area: float):
        self.area = area

class BasicCell(Cell):
    def step(self, cells: list, annealing: bool = False):
        if annealing:
            net_force = self.calculate_attraction(cells)
        else:
            net_force = self.calculate_attraction(cells) - self.calculate_repulsion(cells)

        self.x, self.y = [self.x, self.y] + 0.95 * 0.005 * net_force


class MulticiliatedCell(Cell):
    def step(self, cells: list, annealing: bool = False):
        if annealing:
            net_force = self.calculate_attraction(cells)
        else:
            net_force = self.calculate_attraction(cells) - self.calculate_repulsion(cells)

        self.x, self.y = [self.x, self.y] + 0.95 * 0.005 * net_force

    def external_force(self, direction: np.ndarray, magnitude: float):
        force = (direction * magnitude) / np.linalg.norm(direction) 
        self.x, self.y = [self.x, self.y] + 0.95 * 0.005 * force

class BorderCell(Cell):
    def step(self, cells: list, annealing: bool = False):
        pass
