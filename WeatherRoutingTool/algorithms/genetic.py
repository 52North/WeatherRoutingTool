import logging
from datetime import timedelta

import numpy as np
import matplotlib
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize

import WeatherRoutingTool.utils.formatting as form
from WeatherRoutingTool.algorithms.routingalg import RoutingAlg
from WeatherRoutingTool.algorithms.data_utils import distance, find_start_and_end, load_data, time_diffs
from WeatherRoutingTool.algorithms.genetic_utils import (GeneticCrossover, GeneticMutation, PopulationFactory,
                                                         RoutingProblem)
from WeatherRoutingTool.constraints.constraints import ConstraintsList
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.ship.ship import Boat
from WeatherRoutingTool.utils.maps import Map
from WeatherRoutingTool.weather import WeatherCond

logger = logging.getLogger('WRT.Genetic')


class Genetic(RoutingAlg):
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

        self.ncount = config.GENETIC_NUMBER_GENERATIONS   # ToDo: use better name than ncount?
        self.count = 0

        self.n_offsprings = config.GENETIC_NUMBER_OFFSPRINGS
        self.pop_size = config.GENETIC_POPULATION_SIZE
        self.population_type = config.GENETIC_POPULATION_TYPE

        self.ship_params = None

        self.print_init()

    def execute_routing(self, boat: Boat, wt: WeatherCond, constraints_list: ConstraintsList, verbose=False):
        data = load_data(self.weather_path)
        wave_height = data.VHM0.isel(time=0)
        problem = RoutingProblem(departure_time=self.departure_time, boat=boat, grid_points=wave_height,
                                 constraint_list=constraints_list)
        problem.interpolate_grid(10, 10)
        # ToDo: set start and end in __init__ and use self.start and self.end
        start, end = find_start_and_end(self.start[0], self.start[1], self.finish[0], self.finish[1],
                                        problem.get_grid())
        # ToDo: add factories for GeneticCrossover, GeneticMutation and RoutingProblem
        initial_population = PopulationFactory.get_population(self.population_type, start, end, grid_points=wave_height)
        crossover = GeneticCrossover()
        mutation = GeneticMutation(wave_height)
        res = self.optimize(problem, initial_population, crossover, mutation)
        # get the best solution
        best_idx = res.F.argmin()
        best_x = res.X[best_idx]
        # best_f = res.F[best_idx]
        route = best_x[0]
        self.route = route
        _, self.ship_params = problem.get_power([route])
        result = self.terminate(problem)
        # print(route)
        # print(result)
        return result

    def print_init(self):
        logger.info("Initializing Routing......")
        logger.info('route from ' + str(self.start) + ' to ' + str(self.finish))
        # logger.info('start time ' + str(self.time))
        logger.info(form.get_log_step('route from ' + str(self.start) + ' to ' + str(self.finish), 1))
        # logger.info(form.get_log_step('start time ' + str(self.time), 1))

    def print_current_status(self):
        logger.info("ALGORITHM SETTINGS:")
        logger.info('start : ' + str(self.start))
        logger.info('finish : ' + str(self.finish))
        logger.info('generations: ' + str(self.ncount))
        logger.info('pop_size: ' + str(self.pop_size))
        logger.info('offsprings: ' + str(self.n_offsprings))

    # TODO: adjust terminate function to those of the base class
    def terminate(self, problem):
        form.print_line()
        logger.info('Terminating...')

        lats, lons, route = problem.index_to_coords(self.route)
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

    def optimize(self, problem, initial_population, crossover, mutation):
        # cost[nan_mask] = 20000000000* np.nanmax(cost) if np.nanmax(cost) else 0
        algorithm = NSGA2(pop_size=self.pop_size,
                          sampling=initial_population,
                          crossover=crossover,
                          n_offsprings=self.n_offsprings,
                          mutation=mutation,
                          eliminate_duplicates=False,
                          return_least_infeasible=False)
        termination = get_termination("n_gen", self.ncount)

        res = minimize(problem,
                       algorithm,
                       termination,
                       save_history=True,
                       verbose=True)
        # stop = timeit.default_timer()
        # route_cost(res.X)
        return res
