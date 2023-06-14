import datetime as dt
import os
os.chdir('/home/parichay/Mari/MariGeoRoute/Isochrone')
print(os.getcwd())
import config
from algorithms.isofuel import IsoFuel
from algorithms.RunGenetic import RunGenetic

class RoutingAlgFactory():

    def __init__(self):
        pass

    def get_routing_alg(self, alg_type):
        ra = None

        r_la1, r_lo1, r_la2, r_lo2 = config.DEFAULT_ROUTE
        start = (r_la1, r_lo1)
        finish = (r_la2, r_lo2)
        start_time = dt.datetime.strptime(config.START_TIME, '%Y%m%d%H')
        delta_fuel = config.DELTA_FUEL
        fig_path = config.FIGURE_PATH
        routing_steps = config.ROUTING_STEPS

        if alg_type=='ISOFUEL':
            ra = IsoFuel(start, finish, start_time, delta_fuel, fig_path)
            ra.set_steps(routing_steps)
            ra.set_pruning_settings(config.ISOCHRONE_PRUNE_SECTOR_DEG_HALF, config.ISOCHRONE_PRUNE_SEGMENTS)
            ra.set_variant_segments(config.ROUTER_HDGS_SEGMENTS, config.ROUTER_HDGS_INCREMENTS_DEG)

        if alg_type == 'GENETIC':
            ra = RunGenetic(start, finish, start_time, "")

        return ra
