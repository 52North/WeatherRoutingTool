import logging
import os
import time
from datetime import timedelta

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astropy import units as u
from matplotlib.ticker import ScalarFormatter
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.result import Result
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.running_metric import RunningMetric

import WeatherRoutingTool.utils.formatting as formatting
import WeatherRoutingTool.utils.graphics as graphics
from WeatherRoutingTool.algorithms.genetic.population import PopulationFactory
from WeatherRoutingTool.algorithms.genetic.crossover import CrossoverFactory
from WeatherRoutingTool.algorithms.genetic.mutation import MutationFactory
from WeatherRoutingTool.algorithms.genetic.repair import RepairFactory
from WeatherRoutingTool.algorithms.genetic.problem import RoutingProblem
from WeatherRoutingTool.algorithms.genetic import utils

from WeatherRoutingTool.algorithms.routingalg import RoutingAlg
from WeatherRoutingTool.constraints.constraints import ConstraintsList
from WeatherRoutingTool.config import Config
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.ship.ship import Boat
from WeatherRoutingTool.utils.maps import Map
from WeatherRoutingTool.weather import WeatherCond

logger = logging.getLogger("WRT.genetic")


class Genetic(RoutingAlg):
    """Genetic Algorithm implementation for Weather Routing Tool"""

    def __init__(self, config: Config):
        super().__init__(config)

        self.config = config

        # running
        self.figure_path = graphics.get_figure_path()
        if self.figure_path is not None:
            os.makedirs(self.figure_path, exist_ok=True)
        self.default_map: Map = Map(*config.DEFAULT_MAP)

        self.n_generations = config.GENETIC_NUMBER_GENERATIONS
        self.n_offsprings = config.GENETIC_NUMBER_OFFSPRINGS
        self.objectives = config.GENETIC_OBJECTIVES
        self.n_objs = len(config.GENETIC_OBJECTIVES)
        self.get_objective_weights()

        # population
        self.pop_type = config.GENETIC_POPULATION_TYPE
        self.pop_size = config.GENETIC_POPULATION_SIZE

    def get_objective_weights(self):
        self.objective_weights={
            "arrival_time": -99.,
            "fuel_consumption": -99.
        }

        self.objective_weights["arrival_time"] = utils.get_weigths_from_rankarr(
            np.array([self.objectives["arrival_time"]]),
            self.n_objs
        )
        self.objective_weights["fuel_consumption"] = utils.get_weigths_from_rankarr(
            np.array([self.objectives["fuel_consumption"]]),
            self.n_objs
        )



    def execute_routing(
            self,
            boat: Boat,
            wt: WeatherCond,
            constraints_list: ConstraintsList,
            verbose=False
    ):
        """Main routing execution function

        :param boat: Boat config
        :type boat: Boat
        :param wt: Weather conditions for the waypoints
        :type wt: WeatherCond
        :param constraints_list: List of problem constraints
        :type constraints_list: ConstraintsList
        :param verbose: Verbosity setting for logs
        :type verbose: Optional[bool]
        """

        plt.set_loglevel(level='warning')  # deactivate matplotlib debug messages if debug mode activated
        if self.config.GENETIC_FIX_RANDOM_SEED:
            logger.info('Fixing random seed for genetic algorithm.')
            np.random.seed(1)

        # inputs
        problem = RoutingProblem(
            departure_time=self.departure_time,
            arrival_time=self.arrival_time,
            boat_speed=self.boat_speed,
            boat=boat,
            constraint_list=constraints_list,
            objectives=self.objectives
        )

        initial_population = PopulationFactory.get_population(
            self.config, boat, constraints_list, wt, )

        crossover = CrossoverFactory.get_crossover(self.config, constraints_list)

        mutation = MutationFactory.get_mutation(self.config, constraints_list)

        repair = RepairFactory.get_repair(
            self.config, constraints_list)

        duplicates = utils.RouteDuplicateElimination()

        # optimize
        res_minimize = self.optimize(
            problem, initial_population, crossover, mutation, duplicates, repair)

        # terminate
        res_terminate = self.terminate(
            res=res_minimize,
            problem=problem, )

        return res_terminate, 9

    def optimize(
            self,
            problem,
            initial_population,
            crossover,
            mutation,
            duplicates,
            repair,
    ):
        """Optimization function for the Genetic Algorithm"""

        algorithm = NSGA2(
            pop_size=self.pop_size,
            n_offsprings=self.n_offsprings,
            sampling=initial_population,
            crossover=crossover,
            mutation=mutation,
            repair=repair,
            eliminate_duplicates=duplicates,
            return_least_infeasible=False,
        )

        termination = get_termination("n_gen", self.n_generations)

        start_time = time.time()

        algorithm.setup(
            problem=problem,
            algorithm=algorithm,
            termination=termination,
            save_history=True,
            verbose=True, )

        while algorithm.has_next():
            algorithm.next()

        # print statistics
        res = algorithm.result()
        algorithm.mating.crossover.print_crossover_statistics()
        algorithm.mating.mutation.print_mutation_statistics()
        logger.info('Time after minimisation: ' + str(formatting.get_current_time(start_time)))

        return res

    def rank_solutions(self, obj, dec = False):
        rank_ind = np.argsort(obj)
        if dec:
            rank_ind = rank_ind[::-1]
        rank = np.argsort(rank_ind)
        rank = rank + 1
        return rank

    def get_composite_weight(self, pd_table):
        sol_weight_time = pd_table['time_weight'].to_numpy()
        sol_weight_fuel = pd_table['fuel_weight'].to_numpy()
        obj_weight_time = self.objective_weights["arrival_time"]
        obj_weight_fuel = self.objective_weights["fuel_consumption"]

        denominator = np.abs(1./obj_weight_time * sol_weight_time - 1./obj_weight_fuel * sol_weight_fuel) + 0.2
        summand_time = sol_weight_time/denominator * obj_weight_time*obj_weight_time
        summand_fuel = sol_weight_fuel/denominator * obj_weight_fuel*obj_weight_fuel

        composite_weight = sol_weight_time*sol_weight_fuel + summand_time + summand_fuel

        return composite_weight


    def get_best_compromise(self, solutions):
        debug = True

        if debug:
            print('solutions: ', solutions)
            print('solutions shape: ', solutions.shape)

        rmethod_table = pd.DataFrame()

        if debug:
            print('rmethod table: ', rmethod_table)
        rmethod_table['time_obj'] = solutions[:, 0]
        rmethod_table['fuel_obj'] = solutions[:, 1]
        rmethod_table['time_rank'] = self.rank_solutions(solutions[:, 0])
        rmethod_table['fuel_rank'] = self.rank_solutions(solutions[:, 1])
        rmethod_table['time_weight'] = utils.get_weigths_from_rankarr(rmethod_table['time_rank'].to_numpy(), len(solutions))
        rmethod_table['fuel_weight'] = utils.get_weigths_from_rankarr(rmethod_table['fuel_rank'].to_numpy(), len(solutions))
        rmethod_table['composite_weight'] = self.get_composite_weight(rmethod_table)
        rmethod_table['composite_rank'] = self.rank_solutions(rmethod_table['composite_weight'], True)
        best_ind = np.argmax(rmethod_table['composite_rank'].to_numpy())

        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.precision', 3,
                               ):
            print(rmethod_table)
        return best_ind


    def terminate(self, res: Result, problem: RoutingProblem):
        """Genetic Algorithm termination procedures"""

        super().terminate()
        best_index = self.get_best_compromise(res.F)
        best_route = np.atleast_2d(res.X)[best_index, 0]

        fuel_dict = problem.get_power(best_route)
        fuel = fuel_dict["fuel_sum"]
        ship_params=fuel_dict["shipparams"]
        logger.info(f"Best fuel: {fuel}")

        if self.figure_path is not None:
            logger.info(f"Writing figures to {self.figure_path}")

            self.plot_running_metric(res)
            self.plot_population_per_generation(res, best_route)
            self.plot_convergence(res)
            self.plot_coverage(res, best_route)
            self.plot_objective_space(res, best_index)

        lats = best_route[:, 0]
        lons = best_route[:, 1]
        npoints = lats.size - 1
        speed, *_ = ship_params.get_speed()

        waypoint_coords = RouteParams.get_per_waypoint_coords(
            route_lons=lons,
            route_lats=lats,
            start_time=self.departure_time,
            bs=speed, )

        dists = waypoint_coords['dist']
        courses = waypoint_coords['courses']
        start_times = waypoint_coords['start_times']
        travel_times = waypoint_coords['travel_times']
        arrival_time = start_times[-1] + timedelta(seconds=dists[-1].value / speed.value)

        dists = np.append(dists, -99 * u.meter)
        courses = np.append(courses, -99 * u.degree)
        start_times = np.append(start_times, arrival_time)
        travel_times = np.append(travel_times, -99 * u.second)

        route = RouteParams(
            count=npoints - 1,
            start=self.start,
            finish=self.finish,
            gcr=None,
            route_type='min_fuel_route',
            time=travel_times[-1],
            lats_per_step=lats,
            lons_per_step=lons,
            course_per_step=courses[-1],
            dists_per_step=dists[-1],
            starttime_per_step=start_times,
            ship_params_per_step=ship_params, )

        self.check_destination()
        self.check_positive_power()
        return route

    def plot_objective_space(self, res, best_index):
        F = res.F
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
        ax.plot(F[best_index, 0], F[best_index, 1], color='red', marker='o')
        ax.set_xlabel('f1', labelpad=10)
        ax.set_ylabel('f2', labelpad=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.title("Objective Space")

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))  # Force scientific notation

        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        plt.savefig(os.path.join(self.figure_path, 'genetic_objective_space.png'))
        plt.cla()
        plt.close()

    def print_init(self):
        """Log messages to print on algorithm initialization"""

        logger.info("Initializing Routing......")
        logger.info(f"route from {self.start} to {self.finish}")
        logger.info(formatting.get_log_step(
            f"route from {self.start} to {self.finish}", 1))

    def print_current_status(self):
        """Log messages for running status"""

        logger.info("ALGORITHM SETTINGS:")
        logger.info(f"start: {self.start}")
        logger.info(f"finish: {self.finish}")
        logger.info(f"generations: {self.n_generations}")
        logger.info(f"pop_size: {self.pop_size}")
        logger.info(f"n_offsprings: {self.n_offsprings}")

    def plot_running_metric(self, res):
        """Plot running metrics

        :param res: Result object of minimization
        :type res: pymoo.core.result.Result
        """

        running = RunningMetric()

        plt.rcParams['font.size'] = graphics.get_standard('font_size')
        fig, ax = plt.subplots(figsize=graphics.get_standard('fig_size'))

        igen = 0
        delta_nadir = np.full(self.n_generations, -99.)
        delta_ideal = np.full(self.n_generations, -99.)
        for algorithm in res.history:
            running.update(algorithm)

            if igen > 0:
                delta_nadir[igen] = running.delta_nadir[igen - 1]
                delta_ideal[igen] = running.delta_ideal[igen - 1]
            else:
                delta_nadir[igen] = 0
                delta_ideal[igen] = 0

            igen = igen + 1
            if igen == self.n_generations:
                delta_f = running.delta_f
                x_f = (np.arange(len(delta_f)) + 1)

                # plot png
                ax.plot(x_f, delta_f, label="t=%s (*)" % (igen + 1), alpha=0.9, linewidth=3)

                # write to csv
                graphics.write_graph_to_csv(os.path.join(self.figure_path, 'genetic_algorithm_running_metric.csv'), x_f,
                                            delta_f)

        ax.set_yscale("symlog")
        ax.legend()

        ax.set_xlabel("Generation")
        ax.set_ylabel("Δf", rotation=0)
        plt.savefig(os.path.join(self.figure_path, 'genetic_algorithm_running_metric.png'))
        plt.cla()
        plt.close()

        fig, ax = plt.subplots(figsize=graphics.get_standard('fig_size'))
        x_ni = np.arange(self.n_generations)
        ax.plot(x_ni, delta_nadir)
        plt.savefig(os.path.join(self.figure_path, 'genetic_algorithm_delta_nadir.png'))
        plt.cla()
        plt.close()

        fig, ax = plt.subplots(figsize=graphics.get_standard('fig_size'))
        ax.plot(x_ni, delta_ideal)
        plt.savefig(os.path.join(self.figure_path, 'genetic_algorithm_delta_ideal.png'))
        plt.cla()
        plt.close()

    def plot_population_per_generation(self, res, best_route):
        """Plot figures and save them in WRT_FIGURE_PATH

        :param res: Result of GA minimization
        :type res: pymoo.core.result.Result
        :param best_route: Optimum route
        :type best_route: np.ndarray
        """
        input_crs = ccrs.PlateCarree()
        history = res.history
        fig, ax = plt.subplots(figsize=graphics.get_standard('fig_size'))

        for igen in range(len(history)):
            plt.rcParams['font.size'] = graphics.get_standard('font_size')
            figtitlestr = 'Population of Generation ' + str(igen + 1)

            ax.remove()

            fig, ax = graphics.generate_basemap(
                map=self.default_map.get_var_tuple(),
                depth=None,
                start=self.start,
                finish=self.finish,
                title=figtitlestr,
                show_depth=False, )

            last_pop = history[igen].pop.get('X')

            marker_kw = dict(
                marker="o",
                markersize=3,
                markerfacecolor="gold",
                markeredgecolor="black", )

            for iroute in range(0, last_pop.shape[0]):
                if iroute == 0:
                    ax.plot(
                        last_pop[iroute, 0][:, 1],
                        last_pop[iroute, 0][:, 0],
                        **(marker_kw if igen != self.n_generations - 1 else {}),
                        color="firebrick",
                        label=f"full population [{last_pop.shape[0]}]",
                        transform=input_crs)

                else:
                    ax.plot(
                        last_pop[iroute, 0][:, 1],
                        last_pop[iroute, 0][:, 0],
                        **(marker_kw if igen != self.n_generations - 1 else {}),
                        color="firebrick",
                        transform=input_crs)

            if igen == (self.n_generations - 1):
                ax.plot(
                    best_route[:, 1],
                    best_route[:, 0],
                    **marker_kw,
                    color="blue",
                    label="best route",
                    transform=input_crs
                )

            ax.legend()

            figname = f"genetic_algorithm_generation {igen:02}.png"
            plt.savefig(os.path.join(self.figure_path, figname))

    def plot_coverage(self, res, best_route):
        history = res.history
        input_crs = ccrs.PlateCarree()

        # Create an empty plot
        fig, ax = graphics.generate_basemap(
            map=self.default_map.get_var_tuple(),
            depth=None,
            start=self.start,
            finish=self.finish,
            show_depth=False, )

        for igen in range(len(history)):
            last_pop = history[igen].pop.get('X')

            for iroute in range(0, last_pop.shape[0]):
                lats = last_pop[iroute, 0][:, 0]
                lons = last_pop[iroute, 0][:, 1]
                ax.plot(lons, lats, color="blue", linestyle='-', linewidth=1, transform=input_crs, alpha=0.2)

        ax.plot(best_route[:, 1], best_route[:, 0], color="red", label="best route", transform=input_crs)
        legend = plt.legend(title="routes", loc="upper left")
        legend.get_frame().set_alpha(1)

        figname = "spatial_coverage.png"
        plt.savefig(os.path.join(self.figure_path, figname))

    def plot_convergence(self, res):
        """Plot the convergence curve (best objective value per generation)."""

        best_f = []

        for algorithm in res.history:
            # For single-objective, take min of F; for multi-objective, take min of first objective
            F = algorithm.pop.get('F')
            if F.ndim == 2:
                best_f.append(np.min(F[:, 0]))
            else:
                best_f.append(np.min(F))

        n_gen = np.arange(1, len(best_f) + 1)

        # plot png
        plt.figure(figsize=graphics.get_standard('fig_size'))
        plt.plot(n_gen, best_f, marker='o')
        plt.xlabel('Generation')
        plt.ylabel('Best Objective Value')
        plt.title('Convergence Plot')
        plt.grid(True)
        plt.savefig(os.path.join(self.figure_path, 'genetic_algorithm_convergence.png'))
        plt.cla()
        plt.close()

        # write to csv
        graphics.write_graph_to_csv(os.path.join(self.figure_path, 'genetic_algorithm_convergence.csv'), n_gen, best_f)
