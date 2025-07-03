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


class ShuffleMutation(GridMixin, Mutation):
    def __init__(self, grid, prob=0.4):
        super().__init__(grid=grid)

        self.prob = prob

    def _do(self, problem, X, **kw):
        for i in range(X.shape[0]):
            x = X[i, 0]

            cp1, cp2 = sorted(random.sample(range(1, x.shape[0]-1), 2))
            np.random.shuffle(x[cp1:cp2])

            X[i, 0] = x

        return X


class MutationFactory:

    @staticmethod
    def get_mutation(mutation_type: str, grid=None) -> Mutation:
        return ShuffleMutation(grid)

        if mutation_type == 'grid_based':
            mutation = GridBasedMutation(grid)
        else:
            msg = f"Mutation type '{mutation_type}' is invalid!"
            logger.error(msg)
            raise ValueError(msg)
        return mutation
