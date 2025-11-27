import logging

import WeatherRoutingTool.utils.formatting as form
from WeatherRoutingTool.ship.direct_power_boat import DirectPowerBoat
from WeatherRoutingTool.ship.ship import ConstantFuelBoat
from WeatherRoutingTool.ship.maripower_tanker import MariPowerTanker

logger = logging.getLogger('WRT')


class ShipFactory:

    @staticmethod
    def get_ship(config):
        ship = None

        form.print_line()
        logger.info("Initialising of ship type")

        if config.BOAT_TYPE == 'speedy_isobased':
            logger.info('Use speedy isobased model for modeling fuel consumption.')
            ship = ConstantFuelBoat(file_name=config.CONFIG_PATH)
        if config.BOAT_TYPE == 'direct_power_method':
            logger.info('Use direct power method for modeling fuel consumption.')
            ship = DirectPowerBoat(file_name=config.CONFIG_PATH)
        if config.BOAT_TYPE == 'CBT':
            logger.info('Use maripower for modeling fuel consumption.')
            ship = MariPowerTanker(file_name=config.CONFIG_PATH)
        if config.BOAT_TYPE == 'SAL':
            raise NotImplementedError('Ship type SAL is not yet supported!')

        if not ship:
            raise NotImplementedError('The ship type "' + str(config.SHIP_TYPE) + '", that you requested is '
                                                                                  'not implemented.')
        ship.load_data()
        ship.check_data_meaningful()
        ship.print_init()
        return ship
