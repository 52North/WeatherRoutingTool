from datetime import timedelta

import numpy as np
import matplotlib

from WeatherRoutingTool.algorithms.routingalg import RoutingAlg
from WeatherRoutingTool.algorithms.data_utils import distance, find_start_and_end, load_data, time_diffs
from WeatherRoutingTool.algorithms.genetic import optimize
from WeatherRoutingTool.algorithms.genetic_utils import GeneticUtils
from WeatherRoutingTool.constraints.constraints import ConstraintsList
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.ship.ship import Boat
from WeatherRoutingTool.utils.maps import Map
from WeatherRoutingTool.weather import WeatherCond


class RunGenetic(RoutingAlg):
    fig: matplotlib.figure
    route_ensemble: list
    route: np.array  # temp

    pop_size: int
    n_offsprings: int

    default_map: Map
    weather_path: str

    def __init__(self, config) -> None:
        super().__init__(config)

        self.default_map = config.DEFAULT_MAP
        self.weather_path = config.WEATHER_DATA

        self.ncount = 20  # config.N_GEN
        self.count = 0

        self.pop_size = 20  # config.POP_SIZE
        self.n_offsprings = 2  # config.N_OFFSPRINGS

        self.ship_params = None

        self.print_init()

    def execute_routing(self, boat: Boat, wt: WeatherCond, constraints_list: ConstraintsList, verbose=False):
        data = load_data(self.weather_path)
        wave_height = data.VHM0.isel(time=0)
        genetic_util = GeneticUtils(departure_time=self.departure_time, boat=boat, grid_points=wave_height,
                                    constraint_list=constraints_list)
        genetic_util.interpolate_grid(10, 10)
        start, end = find_start_and_end(self.start[0], self.start[1], self.finish[0], self.finish[1],
                                        genetic_util.get_grid())
        res = optimize(start, end, self.pop_size, self.ncount, self.n_offsprings, genetic_util)
        # get the best solution
        best_idx = res.F.argmin()
        best_x = res.X[best_idx]
        best_f = res.F[best_idx]
        route = best_x[0]
        self.route = route
        _, self.ship_params = genetic_util.get_power([route], wave_height, self.departure_time)
        result = self.terminate(genetic_util)
        # print(route)
        # print(result)
        return result

    def print_init(self):
        print("Initializing Routing......")
        print('route from ' + str(self.start) + ' to ' + str(self.finish))
        # print('start time ' + str(self.time))
        logger.info(form.get_log_step('route from ' + str(self.start) + ' to ' + str(self.finish), 1))
        # logger.info(form.get_log_step('start time ' + str(self.time), 1))

    def print_current_status(self):
        print("ALGORITHM SETTINGS:")
        print('start : ' + str(self.start))
        print('finish : ' + str(self.finish))
        print('generations: ' + str(self.ncount))
        print('pop_size: ' + str(self.pop_size))
        print('offsprings: ' + str(self.n_offsprings))

    # TODO: adjust terminate function to those of the base class
    def terminate(self, genetic_util):
        form.print_line()
        print('Terminating...')

        lats, lons, route = genetic_util.index_to_coords(self.route)
        dists = distance(route)
        speed = self.ship_params.get_speed()[0]
        diffs = time_diffs(speed, route)
        # ship_params = get_power()
        self.count = len(lats)

        dt = self.departure_time
        time = np.array([dt]*len(lats))
        times = np.array([t + timedelta(seconds=delta) for t, delta in zip(time, diffs)])

        route = RouteParams(
            count=self.count-3,
            start=self.start,
            finish=self.finish,
            gcr=np.sum(dists),
            route_type='min_time_route',
            time=diffs,  # time diffs
            lats_per_step=lats.to_numpy(),
            lons_per_step=lons.to_numpy(),
            azimuths_per_step=np.zeros(778),
            dists_per_step=dists,  # dist of waypoints
            starttime_per_step=times,  # time for each point
            ship_params_per_step=self.ship_params
        )

        self.check_destination()
        self.check_positive_power()
        return route

    def check_destination(self):
        pass

    def check_positive_power(self):
        pass
