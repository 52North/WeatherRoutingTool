import logging

import WeatherRoutingTool.utils.formatting as form
from WeatherRoutingTool.ship.direct_power_boat import DirectPowerBoat
from WeatherRoutingTool.ship.ship import ConstantFuelBoat
from WeatherRoutingTool.ship.maripower_tanker import MariPowerTanker

logger = logging.getLogger("WRT")


class ShipFactory:

    @staticmethod
    def get_ship(boat_type: str, ship_config):
        form.print_line()
        logger.info("Initialising ship")

        # Built-in ship models
        if boat_type == "speedy_isobased":
            logger.info("Use speedy isobased model for modeling fuel consumption.")
            ship = ConstantFuelBoat(ship_config)
        elif boat_type == "direct_power_method":
            logger.info("Use direct power method for modeling fuel consumption.")
            ship = DirectPowerBoat(ship_config)
        # ---
        # DEPRECATED: should be removed once the ships are available in the new plugin structure
        elif boat_type == "CBT":
            logger.info("Use maripower for modeling fuel consumption.")
            ship = MariPowerTanker(ship_config)
        elif boat_type == "SAL":
            raise NotImplementedError("Ship type SAL is not yet supported!")
        # ---

        # Ship models dynamically provided via Entry Points
        # https://setuptools.pypa.io/en/latest/userguide/entry_point.html
        else:
            from importlib.metadata import entry_points
            ship_entry_points = entry_points(group="WeatherRoutingTool.ship.ship.Boat")
            try:
                ship_class = ship_entry_points[boat_type].load()
                ship = ship_class(ship_config)
            except KeyError:
                raise NotImplementedError(f"Could not load ship type '{boat_type}' from any plugin.")

        if not ship:
            raise NotImplementedError(f"The ship type '{boat_type}' is not available.")
        ship.load_data()
        ship.check_data_meaningful()
        ship.print_init()
        return ship
