import logging

import WeatherRoutingTool.utils.formatting as form
from WeatherRoutingTool.algorithms.genetic import Genetic
from WeatherRoutingTool.algorithms.isofuel import IsoFuel

logger = logging.getLogger('WRT')


class RoutingAlgFactory:
    """
    Class for generating a routing algorithm object based on the ALGORITHM_TYPE defined in the config
    """

    @staticmethod
    def get_routing_alg(config):
        ra = None
        form.print_line()
        logger.info("Initialising and starting routing procedure. For log output check the files 'info.log' and "
                    "'performance.log'.")

        if (config.ALGORITHM_TYPE == 'isofuel') or (config.ALGORITHM_TYPE == 'speedy_isobased'):
            ra = IsoFuel(config)

        if config.ALGORITHM_TYPE == 'genetic':
            ra = Genetic(config)

        ra.print_init()
        return ra
