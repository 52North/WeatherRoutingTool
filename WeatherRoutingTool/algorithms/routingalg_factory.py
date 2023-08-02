import datetime as dt

import WeatherRoutingTool.config as config
from WeatherRoutingTool.algorithms.isofuel import IsoFuel


class RoutingAlgFactory():

    def __init__(self):
        pass

    def get_routing_alg(self, alg_type):
        ra = None

        lat_start, lon_start, lat_end, lon_end = config.DEFAULT_ROUTE
        start = (lat_start, lon_start)
        finish = (lat_end, lon_end)
        start_time = dt.datetime.strptime(config.START_TIME, '%Y%m%d%H')
        delta_fuel = config.DELTA_FUEL
        fig_path = config.FIGURE_PATH
        routing_steps = config.ROUTING_STEPS

        if alg_type == 'ISOFUEL':
            ra = IsoFuel(start, finish, start_time, delta_fuel, fig_path)
            ra.set_steps(routing_steps)
            ra.set_pruning_settings(config.ISOCHRONE_PRUNE_SECTOR_DEG_HALF, config.ISOCHRONE_PRUNE_SEGMENTS)
            ra.set_variant_segments(config.ROUTER_HDGS_SEGMENTS, config.ROUTER_HDGS_INCREMENTS_DEG)

        return ra
