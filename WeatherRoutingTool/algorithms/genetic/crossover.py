import json
import logging
import os
import random
from math import ceil
from pathlib import Path
from datetime import datetime

import numpy as np
from geographiclib.geodesic import Geodesic
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.repair import Repair
from skimage.graph import route_through_array

from WeatherRoutingTool.ship.ship import Boat
from WeatherRoutingTool.algorithms.data_utils import GridMixin
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.utils.graphics import plot_genetic_algorithm_initial_population

logger = logging.getLogger('WRT.Genetic')


class SinglePointCrossover(Crossover):
    """
    Custom class to define genetic crossover for routes
    """
    def __init__(self, prob=1):

        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, _ = X.shape
        Y = np.full_like(X, None, dtype=object)
        for k in range(n_matings):
            # get the first and the second parent
            a, b = X[0, k, 0], X[1, k, 0]
            Y[0, k, 0], Y[1, k, 0] = self.crossover(a, b)
        return Y

    def crossover(self, parent1, parent2):
        # src = parent1[0]
        # dest = parent1[-1]
        intersect = np.array([x for x in parent1 if any((x == y).all() for y in parent2)])

        if len(intersect) == 0:
            return parent1, parent2
        else:
            cross_over_point = random.choice(intersect)
            idx1 = np.where((parent1 == cross_over_point).all(axis=1))[0][0]
            idx2 = np.where((parent2 == cross_over_point).all(axis=1))[0][0]
            child1 = np.concatenate((parent1[:idx1], parent2[idx2:]), axis=0)
            child2 = np.concatenate((parent2[:idx2], parent1[idx1:]), axis=0)  # print(child1, child2)
        return child1, child2


import numpy as np

def impute_n8_path(points):
    """
    Ensure all adjacent points in the array are 8-connected neighbors.
    Insert waypoints between points that are not n8 neighbors.

    Parameters:
    points (numpy.ndarray): Array of shape (n, 2) containing [x, y] coordinates

    Returns:
    numpy.ndarray: Array with interpolated waypoints ensuring n8 connectivity
    """
    if len(points) < 2:
        return points

    result = [points[0]]  # Start with the first point

    for i in range(1, len(points)):
        current = points[i-1]
        next_point = points[i]

        # Calculate the difference between consecutive points
        dx = next_point[0] - current[0]
        dy = next_point[1] - current[1]

        # Check if points are n8 neighbors (max distance of 1 in both x and y)
        if abs(dx) <= 1 and abs(dy) <= 1:
            # Already n8 neighbors, just add the next point
            result.append(next_point)
        else:
            # Need to interpolate waypoints
            waypoints = interpolate_n8_path(current, next_point)
            result.extend(waypoints[1:])  # Skip the first point (already in result)

    return np.array(result)

def interpolate_n8_path(start, end):
    """
    Generate a path of n8-connected waypoints between two points.
    Uses Bresenham-like algorithm optimized for 8-connectivity.

    Parameters:
    start (array-like): Starting point [x, y]
    end (array-like): Ending point [x, y]

    Returns:
    list: List of waypoints including start and end points
    """
    x0, y0 = start
    x1, y1 = end

    path = [(x0, y0)]

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    # Direction of movement
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    x, y = x0, y0

    # Use the maximum of dx and dy as the number of steps
    # This ensures 8-connectivity (diagonal moves allowed)
    steps = max(dx, dy)

    for i in range(steps):
        if i > 0:  # Don't add the starting point again
            path.append((x, y))

        # Calculate next position
        # Move diagonally when possible, otherwise move in the direction with larger error
        next_x = x
        next_y = y

        if abs(x1 - x) > 0:
            next_x = x + sx
        if abs(y1 - y) > 0:
            next_y = y + sy

        x, y = next_x, next_y

    # Ensure we end at the target point
    if (x, y) != (x1, y1):
        path.append((x1, y1))

    return path


class PMX(GridMixin, Crossover):
    """Partially Mapped Crossover"""

    def __init__(self, grid=None, prob=.5):
        super().__init__(grid=grid, n_parents=2, n_offsprings=2, prob=prob,)

    def _do(self, problem, X, **kw):
        n_parents, n_matings, n_var = X.shape

        # return var
        Y = np.full_like(X, None, dtype=object)

        for k in range(n_matings):
            p1, p2 = X[0, k, 0], X[1, k, 0]
            Y[0, k, 0], Y[1, k, 0] = self.pmx_crossover(p1.copy(), p2.copy())

            Y[0, k, 0] = self.impute(Y[0, k, 0])
            Y[1, k, 0] = self.impute(Y[1, k, 0])
        return Y

    def impute(self, pt: np.ndarray):
        *_, indices = self.coords_to_index(pt)
        indices = np.array(indices)
        indices = impute_n8_path(indices)

        *_, rpt = self.index_to_coords(indices)
        rpt = np.array(rpt)
        rpt[[0, -1]] = pt[[0, -1]]

        return rpt

    def pmx_crossover(self, p1: np.ndarray, p2: np.ndarray):
        """Perform Partially Mapped Crossover (PMX) between two parent routes.

        Args:
            p1, p2: np.ndarray of shape (N,2) with the same set of waypoints.

        Returns:
            Two offspring np.ndarrays of shape (N,2).
        """

        if p1.shape != p2.shape:
            logging.info("PMX — Not of equal length")
            return p1, p2

        N = min(p1.shape[0], p2.shape[0])

        # Convert to lists of tuples
        parent1 = [tuple(row) for row in p1]
        parent2 = [tuple(row) for row in p2]

        # Choose crossover points
        cx1, cx2 = sorted(random.sample(range(N), 2))

        # Initialize offspring placeholders
        child1 = [None] * N
        child2 = [None] * N

        # Copy the segment
        for i in range(cx1, cx2):
            child1[i] = parent2[i]
            child2[i] = parent1[i]

        # Build mapping for the swapped segments
        mapping12 = {parent2[i]: parent1[i] for i in range(cx1, cx2)}
        mapping21 = {parent1[i]: parent2[i] for i in range(cx1, cx2)}

        def resolve(gene, segment, mapping):
            # Keep resolving until gene is not in the given segment
            while gene in segment:
                gene = mapping[gene]
            return gene

        # Fill remaining positions
        for i in range(N):
            if not (cx1 <= i < cx2):
                g1 = parent1[i]
                g2 = parent2[i]

                # If g1 is already in the swapped segment of child1, resolve via mapping12
                if g1 in child1[cx1:cx2]:
                    g1 = resolve(g1, child1[cx1:cx2], mapping12)
                child1[i] = g1

                # Likewise for child2
                if g2 in child2[cx1:cx2]:
                    g2 = resolve(g2, child2[cx1:cx2], mapping21)
                child2[i] = g2

        # Convert back to numpy arrays
        c1 = np.array(child1, dtype=p1.dtype)
        c2 = np.array(child2, dtype=p1.dtype)

        return c1, c2


# factory
class CrossoverFactory:

    @staticmethod
    def get_crossover(crossover_type: str, grid=None):
        crossover = PMX(grid=grid)
        return crossover
