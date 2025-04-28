import logging

import WeatherRoutingTool.utils.formatting as form
from WeatherRoutingTool.ship.ship import ConstantFuelBoat, Tanker, DirectPowerBoat

logger = logging.getLogger('WRT')


class ShipFactory:

    @staticmethod
    def get_ship(config):
        ship = None

        logger.info("Initialising of ship type")
        form.print_line()

        if config.ALGORITHM_TYPE == 'speedy_isobased':
            ship = ConstantFuelBoat(config)
        if config.SHIP_TYPE == 'direct_power_method':
            print('Using direct power method')
            ship = DirectPowerBoat(config)
        if config.SHIP_TYPE == 'CBT':
            ship = Tanker(config)
        if config.SHIP_TYPE == 'SAL':
            raise NotImplementedError('Ship type SAL is not yet supported!')

        if not ship:
            raise NotImplementedError('The ship type "' + str(config.SHIP_TYPE) + '", that you requested is '
                                                                                  'not implemented.')

        ship.print_init()
        return ship
