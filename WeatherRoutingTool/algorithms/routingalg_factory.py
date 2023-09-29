import datetime as dt
import WeatherRoutingTool.config as config
import WeatherRoutingTool.utils.formatting as form
from WeatherRoutingTool.algorithms.isofuel import IsoFuel
from WeatherRoutingTool.algorithms.RunGenetic import RunGenetic



class RoutingAlgFactory:

    @classmethod
    def get_routing_alg(cls, alg_type):
        ra = None

        lat_start, lon_start, lat_end, lon_end = config.DEFAULT_ROUTE
        start = (lat_start, lon_start)
        finish = (lat_end, lon_end)
        departure_time = dt.datetime.strptime(config.DEPARTURE_TIME, '%Y-%m-%dT%H:%MZ')
        delta_fuel = config.DELTA_FUEL
        fig_path = config.FIGURE_PATH
        routing_steps = config.ROUTING_STEPS

        print(
            'Initialising and starting routing procedure. For log output check the files "info.log" and '
            '"performance.log".')
        form.print_line()

        if alg_type == 'isofuel':
            ra = IsoFuel(start, finish, departure_time, delta_fuel, fig_path)
            ra.set_steps(routing_steps)
            ra.set_pruning_settings(config.ISOCHRONE_PRUNE_SECTOR_DEG_HALF, config.ISOCHRONE_PRUNE_SEGMENTS)
            ra.set_variant_segments(config.ROUTER_HDGS_SEGMENTS, config.ROUTER_HDGS_INCREMENTS_DEG)

        if alg_type == 'genetic':
            ra = RunGenetic(start, finish, departure_time, "")

        return ra
