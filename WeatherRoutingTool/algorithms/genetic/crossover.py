from pymoo.core.crossover import Crossover

import numpy as np

import logging

from WeatherRoutingTool.config import Config

from WeatherRoutingTool.algorithms.genetic import utils

logger = logging.getLogger("WRT.genetic.crossover")


class NoCrossover(Crossover):
    def _do(self, problem, X, **kw):
        return X


class SinglePointCrossover(Crossover):
    """Single-point crossover operator with great-circle patching"""

    def __init__(self, prob=1.):
        super().__init__(n_parents=2, n_offsprings=2, prob=prob)

    def _do(self, problem, X, **kw):
        n_parents, n_matings, n_var = X.shape

        Y = np.full_like(X, None, dtype=object)

        for k in range(n_matings):
            p1, p2 = X[0, k, 0], X[1, k, 0]
            Y[0, k, 0], Y[1, k, 0] = self.crossover(p1.copy(), p2.copy())
        return Y

    def crossover(self, p1, p2):
        xp1 = np.random.randint(1, p1.shape[0] - 1)
        xp2 = np.random.randint(1, p2.shape[0] - 1)

        gcr_dist = 1e6

        r1 = np.concatenate([
            p1[:xp1],
            utils.great_circle_route(tuple(p1[xp1-1]), tuple(p2[xp2]), distance=gcr_dist),
            p2[xp2:]])

        r2 = np.concatenate([
            p2[:xp2],
            utils.great_circle_route(tuple(p2[xp2-1]), tuple(p1[xp1]), distance=gcr_dist),
            p1[xp1:]])

        return r1, r2


class CrossoverFactory:
    @staticmethod
    def get_crossover(config: Config):
        return SinglePointCrossover(prob=.5)
        return NoCrossover(n_parents=2, n_offsprings=2, prob=.5)
