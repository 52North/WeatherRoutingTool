from pymoo.core.crossover import Crossover

import logging

from WeatherRoutingTool.config import Config

logger = logging.getLogger("WRT.genetic.crossover")


class NoCrossover(Crossover):
    def _do(self, problem, X, **kw):
        return X


class CrossoverFactory:
    @staticmethod
    def get_crossover(config: Config):
        return NoCrossover(n_parents=2, n_offsprings=2)
