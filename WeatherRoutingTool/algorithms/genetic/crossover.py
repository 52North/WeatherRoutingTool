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


class PMX(Crossover):
    """Partially Mapped Crossover"""

    def __init__(self, prob=.5):
        super().__init__(n_parents=2, n_offsprings=2, prob=prob,)

    def _do(self, problem, X, **kw):
        n_parents, n_matings, n_var = X.shape

        # return var
        Y = np.full_like(X, None, dtype=object)

        for k in range(n_matings):
            p1, p2 = X[0, k, 0], X[1, k, 0]
            Y[0, k, 0], Y[1, k, 0] = self.pmx_crossover(p1.copy(), p2.copy())
        return Y

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


class GeneticCrossover_X(Crossover):
    """
    Enhanced genetic crossover for route optimization with multiple strategies
    """
    def __init__(self, prob=1.0, crossover_type='intersection_based', grid=None):
        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)
        self.prob = prob
        self.crossover_type = crossover_type
        self.grid = grid

    def _do(self, problem, X, **kwargs):
        # The input has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape
        Y = np.full_like(X, None, dtype=object)

        for k in range(n_matings):
            # get the first and the second parent
            parent1, parent2 = X[0, k, 0], X[1, k, 0]

            # Apply crossover with probability
            if np.random.random() < self.prob:
                child1, child2 = self.cross_over(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            Y[0, k, 0], Y[1, k, 0] = child1, child2

        return Y

    def cross_over(self, parent1, parent2):
        """
        Main crossover method that delegates to specific crossover strategies
        """
        if self.crossover_type == 'intersection_based':
            return self._intersection_based_crossover(parent1, parent2)
        elif self.crossover_type == 'order_based':
            return self._order_based_crossover(parent1, parent2)
        elif self.crossover_type == 'uniform':
            return self._uniform_crossover(parent1, parent2)
        elif self.crossover_type == 'geometric':
            return self._geometric_crossover(parent1, parent2)
        elif self.crossover_type == 'adaptive':
            return self._adaptive_crossover(parent1, parent2)
        else:
            logger.warning(f"Unknown crossover type: {self.crossover_type}. Using intersection_based.")
            return self._intersection_based_crossover(parent1, parent2)

    def _intersection_based_crossover(self, parent1, parent2):
        """
        Original intersection-based crossover with improvements
        """
        # Ensure start and end points are preserved
        src, dest = parent1[0], parent1[-1]

        # Find intersection points between routes
        intersect = np.array([x for x in parent1 if any((x == y).all() for y in parent2)])

        if len(intersect) == 0:
            # No intersection points - use midpoint crossover
            return self._midpoint_crossover(parent1, parent2)
        else:
            # Use intersection point for crossover
            cross_over_point = random.choice(intersect)
            idx1 = np.where((parent1 == cross_over_point).all(axis=1))[0][0]
            idx2 = np.where((parent2 == cross_over_point).all(axis=1))[0][0]

            child1 = np.concatenate((parent1[:idx1], parent2[idx2:]), axis=0)
            child2 = np.concatenate((parent2[:idx2], parent1[idx1:]), axis=0)

            # Ensure start and end points are correct
            child1[0], child1[-1] = src, dest
            child2[0], child2[-1] = src, dest

            return child1, child2

    def _midpoint_crossover(self, parent1, parent2):
        """
        Crossover at midpoint when no intersection points exist
        """
        src, dest = parent1[0], parent1[-1]

        # Use middle point of each route
        mid1 = len(parent1) // 2
        mid2 = len(parent2) // 2

        child1 = np.concatenate((parent1[:mid1], parent2[mid2:]), axis=0)
        child2 = np.concatenate((parent2[:mid2], parent1[mid1:]), axis=0)

        # Ensure start and end points are correct
        child1[0], child1[-1] = src, dest
        child2[0], child2[-1] = src, dest

        return child1, child2

    def _order_based_crossover(self, parent1, parent2):
        """
        Order-based crossover that preserves relative ordering of waypoints
        """
        src, dest = parent1[0], parent1[-1]

        # Create waypoint ordering based on distance from start
        def get_waypoint_order(route):
            distances = []
            for i, point in enumerate(route[1:-1]):  # Exclude start and end
                dist = np.sqrt((point[0] - src[0])**2 + (point[1] - src[1])**2)
                distances.append((dist, i+1, point))
            return sorted(distances, key=lambda x: x[0])

        order1 = get_waypoint_order(parent1)
        order2 = get_waypoint_order(parent2)

        # Create children by alternating between parent orders
        child1_waypoints = []
        child2_waypoints = []

        max_len = max(len(order1), len(order2))
        for i in range(max_len):
            if i < len(order1) and i < len(order2):
                if i % 2 == 0:
                    child1_waypoints.append(order1[i][2])
                    child2_waypoints.append(order2[i][2])
                else:
                    child1_waypoints.append(order2[i][2])
                    child2_waypoints.append(order1[i][2])
            elif i < len(order1):
                child1_waypoints.append(order1[i][2])
                child2_waypoints.append(order1[i][2])
            else:
                child1_waypoints.append(order2[i][2])
                child2_waypoints.append(order2[i][2])

        child1 = np.vstack([src, child1_waypoints, dest])
        child2 = np.vstack([src, child2_waypoints, dest])

        return child1, child2

    def _uniform_crossover(self, parent1, parent2):
        """
        Uniform crossover that randomly selects waypoints from either parent
        """
        src, dest = parent1[0], parent1[-1]

        # Ensure both parents have same length for uniform crossover
        min_len = min(len(parent1), len(parent2))

        child1_waypoints = []
        child2_waypoints = []

        for i in range(1, min_len - 1):  # Skip start and end points
            if np.random.random() < 0.5:
                child1_waypoints.append(parent1[i])
                child2_waypoints.append(parent2[i])
            else:
                child1_waypoints.append(parent2[i])
                child2_waypoints.append(parent1[i])

        # Handle remaining waypoints
        if len(parent1) > min_len:
            child1_waypoints.extend(parent1[min_len-1:-1])
            child2_waypoints.extend(parent1[min_len-1:-1])
        elif len(parent2) > min_len:
            child1_waypoints.extend(parent2[min_len-1:-1])
            child2_waypoints.extend(parent2[min_len-1:-1])

        child1 = np.vstack([src, child1_waypoints, dest])
        child2 = np.vstack([src, child2_waypoints, dest])

        return child1, child2

    def _geometric_crossover(self, parent1, parent2):
        """
        Geometric crossover that creates waypoints as weighted averages
        """
        src, dest = parent1[0], parent1[-1]

        # Ensure both parents have same length
        min_len = min(len(parent1), len(parent2))

        child1_waypoints = []
        child2_waypoints = []

        for i in range(1, min_len - 1):  # Skip start and end points
            # Create weighted average of waypoints
            alpha = np.random.random()
            beta = 1 - alpha

            new_point1 = alpha * parent1[i] + beta * parent2[i]
            new_point2 = beta * parent1[i] + alpha * parent2[i]

            child1_waypoints.append(new_point1)
            child2_waypoints.append(new_point2)

        # Handle remaining waypoints
        if len(parent1) > min_len:
            child1_waypoints.extend(parent1[min_len-1:-1])
            child2_waypoints.extend(parent1[min_len-1:-1])
        elif len(parent2) > min_len:
            child1_waypoints.extend(parent2[min_len-1:-1])
            child2_waypoints.extend(parent2[min_len-1:-1])

        child1 = np.vstack([src, child1_waypoints, dest])
        child2 = np.vstack([src, child2_waypoints, dest])

        return child1, child2

    def _adaptive_crossover(self, parent1, parent2):
        """
        Adaptive crossover that chooses strategy based on route characteristics
        """
        # Calculate route characteristics
        len1, len2 = len(parent1), len(parent2)
        intersect_count = len([x for x in parent1 if any((x == y).all() for y in parent2)])

        # Choose crossover strategy based on characteristics
        if intersect_count > 0:
            # Use intersection-based if routes intersect
            return self._intersection_based_crossover(parent1, parent2)
        elif abs(len1 - len2) <= 2:
            # Use uniform crossover if routes are similar length
            return self._uniform_crossover(parent1, parent2)
        else:
            # Use order-based for very different routes
            return self._order_based_crossover(parent1, parent2)

    def _repair_route(self, route, src, dest):
        """
        Repair route to ensure it's valid (start/end points correct, no duplicates)
        """
        # Ensure start and end points
        route[0] = src
        route[-1] = dest

        # Remove duplicate consecutive waypoints
        unique_indices = []
        for i, point in enumerate(route):
            if i == 0 or not np.array_equal(point, route[i-1]):
                unique_indices.append(i)

        return route[unique_indices]


# factory
class CrossoverFactory:

    @staticmethod
    def get_crossover(crossover_type: str):
        crossover = PMX()
        return crossover
