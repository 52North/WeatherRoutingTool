import logging

import WeatherRoutingTool.utils.formatting as form
from WeatherRoutingTool.ship.ship import ConstantFuelBoat, Tanker

logger = logging.getLogger('WRT')


class ShipFactory:

    @classmethod
    def get_ship(cls, config):
        ship = None

        logger.info("Initialising of ship type")
        form.print_line()

        if config.ALGORITHM_TYPE == 'speedy_isobased':
            ship = ConstantFuelBoat(config)
        else:
            if config.SHIP_TYPE == 'CBT':
                ship = Tanker(config)
            if config.SHIP_TYPE == 'SAL':
                raise NotImplementedError('Ship type SAL is not yet supported!')

        ship.print_init()
        return ship
