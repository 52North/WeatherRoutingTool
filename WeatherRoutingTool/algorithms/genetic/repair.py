from pymoo.core.repair import Repair

import logging

from WeatherRoutingTool.config import Config

logger = logging.getLogger("WRT.genetic.repair")


class RepairFactory:
    @staticmethod
    def get_repair(config: Config):
        pass
