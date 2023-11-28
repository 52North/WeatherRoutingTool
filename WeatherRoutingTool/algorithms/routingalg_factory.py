import logging

import WeatherRoutingTool.utils.formatting as form
from WeatherRoutingTool.algorithms.genetic import Genetic
from WeatherRoutingTool.algorithms.isofuel import IsoFuel

logger = logging.getLogger('WRT')


class RoutingAlgFactory:

    @classmethod
    def get_routing_alg(cls, config):
        ra = None

        logger.info("Initialising and starting routing procedure. For log output check the files 'info.log' and "
                    "'performance.log'.")
        form.print_line()

        if config.ALGORITHM_TYPE == 'isofuel':
            ra = IsoFuel(config)

        if config.ALGORITHM_TYPE == 'genetic':
            ra = Genetic(config)

        ra.print_init()
        return ra
