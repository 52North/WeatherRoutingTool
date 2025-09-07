from pymoo.core.mutation import Mutation

import logging

from WeatherRoutingTool.config import Config

logger = logging.getLogger("WRT.genetic.mutation")


class MutationFactory:
    @staticmethod
    def get_mutation(config: Config) -> Mutation:
        return Mutation(prob=1.)
