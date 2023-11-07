import datetime as dt

import WeatherRoutingTool.utils.formatting as form
from WeatherRoutingTool.algorithms.isofuel import IsoFuel


class RoutingAlgFactory:

    @classmethod
    def get_routing_alg(cls, config):
        ra = None

        lat_start, lon_start, lat_end, lon_end = config.DEFAULT_ROUTE
        start = (lat_start, lon_start)
        finish = (lat_end, lon_end)
        departure_time = dt.datetime.strptime(config.DEPARTURE_TIME, '%Y-%m-%dT%H:%MZ')
        delta_fuel = config.DELTA_FUEL
        fig_path = config.FIGURE_PATH
        routing_steps = config.ROUTING_STEPS

        print('Initialising and starting routing procedure. For log output check the files "info.log" and '
              '"performance.log".')
        form.print_line()

        if config.ALGORITHM_TYPE == 'isofuel':
            ra = IsoFuel(start, finish, departure_time, delta_fuel, fig_path)
            ra.set_steps(routing_steps)
            ra.set_pruning_settings(sector_deg_half=config.ISOCHRONE_PRUNE_SECTOR_DEG_HALF,
                                    seg=config.ISOCHRONE_PRUNE_SEGMENTS,
                                    prune_bearings=config.ISOCHRONE_PRUNE_BEARING,
                                    prune_gcr_centered=config.ISOCHRONE_PRUNE_GCR_CENTERED)
            ra.set_variant_segments(config.ROUTER_HDGS_SEGMENTS, config.ROUTER_HDGS_INCREMENTS_DEG)
            ra.set_minimisation_criterion(config.ISOCHRONE_MINIMISATION_CRITERION)

        ra.print_init()

        return ra
