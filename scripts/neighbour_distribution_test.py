from scipy.spatial import Voronoi
from tqdm import tqdm
from collections import defaultdict, Counter
import multiprocessing as mp

from ciliasim import geometry

N = 1000
width = height = 10

def simulate(_):
    neighbours = defaultdict(set)
    layout = geometry.uniform_random_layout(width, height, int(width * height))
    voronoi = Voronoi(layout)

    for p1, p2 in voronoi.ridge_points:
        neighbours[p1].add(p2)
        neighbours[p2].add(p1)

    degrees = [len(neighbours[j]) for j in range(len(layout))]
    distribution = Counter(degrees)

    return distribution, len(layout)

total = Counter()
num_points = 0
with mp.Pool() as pool:
    for distribution, points in tqdm(pool.imap_unordered(simulate, range(N)), total=N):
        total += distribution
        num_points += points

mean_distribution = {k: v / num_points for k, v in total.items()}

print(num_points, mean_distribution)

# Seems like there's a roughly 1 in a million chance of getting a cell with 15 neighbours - N x 15 should be fine
