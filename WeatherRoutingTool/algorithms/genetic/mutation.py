import logging
import copy
import random

import numpy as np
from pymoo.core.mutation import Mutation
from skimage.graph import route_through_array

from WeatherRoutingTool.algorithms.data_utils import GridMixin

logger = logging.getLogger('WRT.Genetic')


class GridBasedMutation(GridMixin, Mutation):
    """
    Custom class to define genetic mutation for routes
    """
    def __init__(self, grid, prob=0.4):
        super().__init__(grid=grid)
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        offsprings = np.zeros((len(X), 1), dtype=object)
        # loop over individuals in population
        for idx, i in enumerate(X):
            # perform mutation with certain probability
            if np.random.uniform(0, 1) < self.prob or True:
                mutated_individual = self.mutate(i[0])
                # print("mutated_individual", mutated_individual, "###")
                offsprings[idx][0] = mutated_individual
            # if no mutation
            else:
                offsprings[idx][0] = i[0]
        return offsprings

    def mutate(self, route):
        size = len(route)
        start = random.randint(1, size - 2)
        end = random.randint(start, size - 2)

        _, _, start_indices = self.coords_to_index([(route[start][0], route[start][1])])
        _, _, end_indices = self.coords_to_index([(route[end][0], route[end][1])])

        shuffled_cost = self.get_shuffled_cost()
        subpath, _ = route_through_array(shuffled_cost, start_indices[0], end_indices[0],
                                         fully_connected=True, geometric=False)
        _, _, subpath = self.index_to_coords(subpath)
        newPath = np.concatenate((route[:start], np.array(subpath), route[end + 1:]), axis=0)
        return newPath


class RandomWalkMutation(GridMixin, Mutation):
    """Mutates a random point in the waypoint to it's N4 alternatives"""

    @staticmethod
    def random_walk(point, to_avoid=None):
        """Randomly pick an N4 neighbour of a waypoint"""

        if to_avoid is None:
            to_avoid = []

        x, y = point

        possible_moves = np.array([
            [x+1, y], [x-1, y],  # [x+1, y+1], [x+1, y-1],
            [x, y+1], [x, y-1],  # [x-1, y+1], [x-1, y-1],
        ])

        possible_moves = [
            x for x in possible_moves
            if not np.any(np.all(to_avoid == x, axis=1))
        ]

        return random.choice(possible_moves)

    def __init__(self, grid, prob=0.4):
        super().__init__(grid=grid)

        self.prob = prob

    def _do(self, problem, X, **kw):
        if not random.random() > self.prob:
            return X

        rs = copy.deepcopy(X)

        for route, in rs:
            rindex = np.random.randint(1, route.shape[0] - 1)

            *_, (point,) = self.coords_to_index(route[[rindex]])
            *_, neighbours = self.coords_to_index(route[[rindex-1, rindex+1]])

            jitter = self.random_walk(point, neighbours)

            *_, (coords,) = self.index_to_coords(jitter[None, :])
            route[rindex] = coords
        return rs


class MutationFactory:
    """
    Factory class for mutation
    """
    @staticmethod
    def get_mutation(mutation_type: str, grid=None) -> Mutation:
        if mutation_type == 'grid_based':
            mutation = RandomWalkMutation(grid)
        else:
            msg = f"Mutation type '{mutation_type}' is invalid!"
            logger.error(msg)
            raise ValueError(msg)
        return mutation
