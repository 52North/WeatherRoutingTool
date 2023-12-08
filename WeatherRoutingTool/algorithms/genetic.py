import logging
import os
from datetime import timedelta

import cartopy.crs as ccrs
import cartopy.feature as cf
import numpy as np
import matplotlib
import xarray as xr
from matplotlib import pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.util.running_metric import RunningMetric

import WeatherRoutingTool.utils.formatting as form
import WeatherRoutingTool.utils.graphics as graphics
from WeatherRoutingTool.algorithms.routingalg import RoutingAlg
from WeatherRoutingTool.algorithms.data_utils import distance, time_diffs
from WeatherRoutingTool.algorithms.genetic_utils import (CrossoverFactory, MutationFactory, PopulationFactory,
                                                         RoutingProblem, RouteDuplicateElimination)
from WeatherRoutingTool.constraints.constraints import ConstraintsList
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.ship.ship import Boat
from WeatherRoutingTool.utils.maps import Map
from WeatherRoutingTool.utils.graphics import get_figure_path
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

        self.ncount = config.GENETIC_NUMBER_GENERATIONS  # ToDo: use better name than ncount?
        self.count = 0
        self.n_offsprings = config.GENETIC_NUMBER_OFFSPRINGS
        self.mutation_type = config.GENETIC_MUTATION_TYPE
        self.pop_size = config.GENETIC_POPULATION_SIZE
        self.population_type = config.GENETIC_POPULATION_TYPE

        self.ship_params = None

        self.print_init()

    def execute_routing(self, boat: Boat, wt: WeatherCond, constraints_list: ConstraintsList, verbose=False):
        data = xr.open_dataset(self.weather_path)
        lat_int, lon_int = 10, 10
        wave_height = data.VHM0.isel(time=0)
        wave_height = wave_height[::lat_int, ::lon_int]
        problem = RoutingProblem(departure_time=self.departure_time, boat=boat, constraint_list=constraints_list)
        initial_population = PopulationFactory.get_population(self.population_type, self.start, self.finish,
                                                              grid=wave_height)
        mutation = MutationFactory.get_mutation(self.mutation_type, grid=wave_height)
        crossover = CrossoverFactory.get_crossover()
        duplicates = RouteDuplicateElimination()
        res = self.optimize(problem, initial_population, crossover, mutation, duplicates)

        result = self.terminate(result_object=res, problem=problem)
        return result

    def print_init(self):
        logger.info("Initializing Routing......")
        logger.info('route from ' + str(self.start) + ' to ' + str(self.finish))
        # logger.info('start time ' + str(self.time))
        logger.info(form.get_log_step('route from ' + str(self.start) + ' to ' + str(self.finish),
                                      1))  # logger.info(form.get_log_step('start time ' + str(self.time), 1))

    def print_current_status(self):
        logger.info("ALGORITHM SETTINGS:")
        logger.info('start : ' + str(self.start))
        logger.info('finish : ' + str(self.finish))
        logger.info('generations: ' + str(self.ncount))
        logger.info('pop_size: ' + str(self.pop_size))
        logger.info('offsprings: ' + str(self.n_offsprings))

    def terminate(self, **kwargs):
        super().terminate()

        res = kwargs.get('result_object')
        problem = kwargs.get('problem')
        figure_path = get_figure_path()

        best_idx = res.F.argmin()
        best_route = res.X[best_idx]
        _, self.ship_params = problem.get_power(best_route)

        if figure_path is not None:
            self.plot_running_metric(res)
            self.plot_population_per_generation(res, best_route)

        lats = best_route[:, 0]
        lons = best_route[:, 1]
        dists = distance(best_route)
        speed = self.ship_params.get_speed()[0]
        diffs = time_diffs(speed, best_route)
        self.count = len(lats)

        dt = self.departure_time
        time = np.array([dt] * len(lats))
        times = np.array([t + timedelta(seconds=delta) for t, delta in zip(time, diffs)])

        route = RouteParams(count=self.count - 3, start=self.start, finish=self.finish, gcr=np.sum(dists),
                            route_type='min_time_route', time=diffs,  # time diffs
                            lats_per_step=lats, lons_per_step=lons,
                            azimuths_per_step=np.zeros(778),
                            dists_per_step=dists,  # dist of waypoints
                            starttime_per_step=times,  # time for each point
                            ship_params_per_step=self.ship_params)

        self.check_destination()
        self.check_positive_power()
        return route

    def check_destination(self):
        pass

    def check_positive_power(self):
        pass

    def optimize(self, problem, initial_population, crossover, mutation, duplicates):
        # cost[nan_mask] = 20000000000* np.nanmax(cost) if np.nanmax(cost) else 0
        algorithm = NSGA2(pop_size=self.pop_size, sampling=initial_population, crossover=crossover,
                          n_offsprings=self.n_offsprings, mutation=mutation, eliminate_duplicates=duplicates,
                          return_least_infeasible=False)
        termination = get_termination("n_gen", self.ncount)

        res = minimize(problem, algorithm, termination, save_history=True, verbose=True)
        # stop = timeit.default_timer()
        # route_cost(res.X)
        return res

    def plot_running_metric(self, res):
        figure_path = get_figure_path()
        running = RunningMetric()

        plt.rcParams['font.size'] = graphics.get_standard('font_size')
        fig, ax = plt.subplots(figsize=graphics.get_standard('fig_size'))

        igen = 0
        delta_nadir = np.full(self.ncount, -99.)
        delta_ideal = np.full(self.ncount, -99.)
        for algorithm in res.history:
            running.update(algorithm)
            delta_f = running.delta_f
            if igen > 0:
                delta_nadir[igen] = running.delta_nadir[igen-1]
                delta_ideal[igen] = running.delta_ideal[igen-1]
            else:
                delta_nadir[igen] = 0
                delta_ideal[igen] = 0

            x_f = (np.arange(len(delta_f)) + 1)
            ax.plot(x_f, delta_f, label="t=%s (*)" % (igen + 1), alpha=0.9, linewidth=3)
            igen = igen + 1
        ax.set_yscale("symlog")
        ax.legend()

        ax.set_xlabel("Generation")
        ax.set_ylabel("Δf", rotation=0)
        plt.savefig(os.path.join(figure_path, 'genetic_algorithm_running_metric.png'))
        plt.cla()
        plt.close()

        fig, ax = plt.subplots(figsize=graphics.get_standard('fig_size'))
        x_ni = np.arange(self.ncount)
        ax.plot(x_ni, delta_nadir)
        plt.savefig(os.path.join(figure_path, 'genetic_algorithm_delta_nadir.png'))
        plt.cla()
        plt.close()

        fig, ax = plt.subplots(figsize=graphics.get_standard('fig_size'))
        ax.plot(x_ni, delta_ideal)
        plt.savefig(os.path.join(figure_path, 'genetic_algorithm_delta_ideal.png'))
        plt.cla()
        plt.close()

    def plot_population_per_generation(self, res, best_route):
        figure_path = get_figure_path()
        history = res.history
        fig, ax = plt.subplots(figsize=graphics.get_standard('fig_size'))

        for igen in range(0, self.ncount):
            plt.rcParams['font.size'] = graphics.get_standard('font_size')
            figtitlestr = 'Population of Generation ' + str(igen + 1)

            ax.remove()
            fig, ax = graphics.generate_basemap(fig, None, self.start, self.finish, figtitlestr, False)

            for iroute in range(0, self.pop_size):
                last_pop = history[igen].pop.get('X')
                if iroute == 0:
                    ax.plot(last_pop[iroute, 0][:, 1], last_pop[iroute, 0][:, 0], color="firebrick",
                            label='full population')
                else:
                    ax.plot(last_pop[iroute, 0][:, 1], last_pop[iroute, 0][:, 0], color="firebrick")
            if igen == (self.ncount - 1):
                ax.plot(best_route[:, 1], best_route[:, 0], color="blue", label='best route')
            ax.legend()
            ax.set_xlim([self.default_map[1], self.default_map[3]])
            ax.set_ylim([self.default_map[0], self.default_map[2]])

            figname = 'genetic_algorithm_generation' + str(igen) + '.png'
            plt.savefig(os.path.join(figure_path, figname))
