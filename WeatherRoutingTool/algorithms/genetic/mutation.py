from pymoo.core.mutation import Mutation

import logging

from WeatherRoutingTool.config import Config

logger = logging.getLogger("WRT.genetic.mutation")


class NoMutation(Mutation):
    def _do(self, problem, X, **kw):
        return X


class MutationFactory:
    @staticmethod
    def get_mutation(config: Config) -> Mutation:
        return NoMutation()
