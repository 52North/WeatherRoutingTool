import logging

import WeatherRoutingTool.utils.formatting as form
from WeatherRoutingTool.ship.direct_power_boat import DirectPowerBoat
from WeatherRoutingTool.ship.ship import ConstantFuelBoat
from WeatherRoutingTool.ship.maripower_tanker import MariPowerTanker

logger = logging.getLogger('WRT')


class ShipFactory:

    @staticmethod
    def get_ship(boat_type: str, ship_config):
        ship = None

        form.print_line()
        logger.info("Initialising ship")

        if boat_type == 'speedy_isobased':
            logger.info('Use speedy isobased model for modeling fuel consumption.')
            ship = ConstantFuelBoat(ship_config)
        if boat_type == 'direct_power_method':
            logger.info('Use direct power method for modeling fuel consumption.')
            ship = DirectPowerBoat(ship_config)
        if boat_type == 'CBT':
            logger.info('Use maripower for modeling fuel consumption.')
            ship = MariPowerTanker(ship_config)
        if boat_type == 'SAL':
            raise NotImplementedError('Ship type SAL is not yet supported!')

        if not ship:
            raise NotImplementedError(f"The ship type '{boat_type}' is not implemented.")
        ship.load_data()
        ship.check_data_meaningful()
        ship.print_init()
        return ship
