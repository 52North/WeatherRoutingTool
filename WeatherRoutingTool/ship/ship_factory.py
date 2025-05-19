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
            logger.info('Use speedy isobased model for modeling fuel consumption.')
            ship = ConstantFuelBoat(file_name=config.CONFIG_PATH)
            if not config.BOAT_TYPE == 'speedy_isobased':
                raise ValueError('The algorithm type "speedy_isobased" can only be applied if the respective is boat '
                                 'type is set correctly, meaning BOAT_TYPE="speedy_isobased".')
        if config.BOAT_TYPE == 'direct_power_method':
            logger.info('Use direct power method for modeling fuel consumption.')
            ship = DirectPowerBoat(file_name=config.CONFIG_PATH)
        if config.BOAT_TYPE == 'CBT':
            logger.info('Use maripower for modeling fuel consumption.')
            ship = Tanker(file_name=config.CONFIG_PATH)
        if config.BOAT_TYPE == 'SAL':
            raise NotImplementedError('Ship type SAL is not yet supported!')

        if not ship:
            raise NotImplementedError('The ship type "' + str(config.SHIP_TYPE) + '", that you requested is '
                                                                                  'not implemented.')
        ship.load_data()
        ship.print_init()
        return ship
