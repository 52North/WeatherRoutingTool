from astropy import units as u
import matplotlib.pyplot as pt
import numpy as np

from datetime import timedelta
import logging
import os

import matplotlib.pyplot as plt
import seaborn as sns

from pymoo.util.running_metric import RunningMetric
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.result import Result

from WeatherRoutingTool.algorithms.routingalg import RoutingAlg
from WeatherRoutingTool.constraints.constraints import ConstraintsList
from WeatherRoutingTool.config import Config
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.ship.ship import Boat
from WeatherRoutingTool.utils.maps import Map
from WeatherRoutingTool.weather import WeatherCond

from WeatherRoutingTool.algorithms.genetic.population import PopulationFactory
from WeatherRoutingTool.algorithms.genetic.crossover import CrossoverFactory
from WeatherRoutingTool.algorithms.genetic.mutation import MutationFactory
from WeatherRoutingTool.algorithms.genetic.repair import RepairFactory
from WeatherRoutingTool.algorithms.genetic.problem import RoutingProblem
from WeatherRoutingTool.algorithms.genetic import utils

import WeatherRoutingTool.utils.formatting as formatting
import WeatherRoutingTool.utils.graphics as graphics

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

        # population
        self.pop_type = config.GENETIC_POPULATION_TYPE
        self.pop_size = config.GENETIC_POPULATION_SIZE

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

        # inputs
        problem = RoutingProblem(
            departure_time=self.departure_time,
            boat=boat,
            constraint_list=constraints_list,)

        initial_population = PopulationFactory.get_population(
            self.config, boat, constraints_list, wt,)

        crossover = CrossoverFactory.get_crossover(
            self.config, constraints_list)

        mutation = MutationFactory.get_mutation(self.config)

        repair = RepairFactory.get_repair(
            self.config, constraints_list)

        duplicates = utils.RouteDuplicateElimination()

        # optimize
        res_minimize = self.optimize(
            problem, initial_population, crossover, mutation, duplicates, repair)

        # terminate
        res_terminate = self.terminate(
            res=res_minimize,
            problem=problem,)

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

        res = minimize(
            problem=problem,
            algorithm=algorithm,
            termination=termination,
            save_history=True,
            verbose=True, )

        return res

    def terminate(self, res: Result, problem: RoutingProblem):
        """Genetic Algorithm termination procedures"""

        super().terminate()

        best_index = res.F.argmin()
        # ensure res.X is of shape (n_sol, n_var)
        best_route = np.atleast_2d(res.X)[best_index, 0]

        fuel, ship_params = problem.get_power(best_route)
        logger.info(f"Best fuel: {fuel}")

        if self.figure_path is not None:
            logger.info(f"Writing figures to {self.figure_path}")

            self.plot_running_metric(res)
            self.plot_population_per_generation(res, best_route)
            self.plot_convergence(res)
            self.plot_coverage(res)

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
        arrival_time = start_times[-1] + \
            timedelta(seconds=dists[-1].value / speed.value)

        dists = np.append(dists, -99 * u.meter)
        courses = np.append(courses, -99 * u.degree)
        start_times = np.append(start_times, arrival_time)
        travel_times = np.append(travel_times, -99 * u.second)

        route = RouteParams(
            count=npoints-1,
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

        pt.rcParams['font.size'] = graphics.get_standard('font_size')
        fig, ax = pt.subplots(figsize=graphics.get_standard('fig_size'))

        igen = 0
        delta_nadir = np.full(self.n_generations, -99.)
        delta_ideal = np.full(self.n_generations, -99.)
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
        pt.savefig(os.path.join(self.figure_path, 'genetic_algorithm_running_metric.png'))
        pt.cla()
        pt.close()

        fig, ax = pt.subplots(figsize=graphics.get_standard('fig_size'))
        x_ni = np.arange(self.n_generations)
        ax.plot(x_ni, delta_nadir)
        pt.savefig(os.path.join(self.figure_path, 'genetic_algorithm_delta_nadir.png'))
        pt.cla()
        pt.close()

        fig, ax = pt.subplots(figsize=graphics.get_standard('fig_size'))
        ax.plot(x_ni, delta_ideal)
        pt.savefig(os.path.join(self.figure_path, 'genetic_algorithm_delta_ideal.png'))
        pt.cla()
        pt.close()

    def plot_population_per_generation(self, res, best_route):
        """Plot figures and save them in WRT_FIGURE_PATH

        :param res: Result of GA minimization
        :type res: pymoo.core.result.Result
        :param best_route: Optimum route
        :type best_route: np.ndarray
        """

        history = res.history
        fig, ax = pt.subplots(figsize=graphics.get_standard('fig_size'))

        for igen in range(len(history)):
            pt.rcParams['font.size'] = graphics.get_standard('font_size')
            figtitlestr = 'Population of Generation ' + str(igen + 1)

            ax.remove()

            fig, ax = graphics.generate_basemap(
                fig=fig,
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
                        label=f"full population [{last_pop.shape[0]}]", )

                else:
                    ax.plot(
                        last_pop[iroute, 0][:, 1],
                        last_pop[iroute, 0][:, 0],
                        **(marker_kw if igen != self.n_generations - 1 else {}),
                        color="firebrick", )

            if igen == (self.n_generations - 1):
                ax.plot(
                    best_route[:, 1],
                    best_route[:, 0],
                    **marker_kw,
                    color="blue",
                    label="best route", )

            ax.legend()
            ax.set_xlim([self.default_map.lon1, self.default_map.lon2])
            ax.set_ylim([self.default_map.lat1, self.default_map.lat2])

            figname = f"genetic_algorithm_generation {igen:02}.png"
            pt.savefig(os.path.join(self.figure_path, figname))

    def plot_coverage(self, res):
        waypoints = None
        history = res.history

        # Create an empty plot
        fig, ax = pt.subplots(figsize=graphics.get_standard('fig_size'))
        fig, ax = graphics.generate_basemap(
            fig=fig,
            depth=None,
            start=self.start,
            finish=self.finish,
            title="spatial coverage",
            show_depth=False, )

        for igen in range(len(history)):
            last_pop = history[igen].pop.get('X')

            for iroute in range(0, last_pop.shape[0]):
                if waypoints is None:
                    waypoints = last_pop[iroute, 0][: , :]
                waypoints = np.concatenate((waypoints, last_pop[iroute, 0][: , :]), axis = 0)

        sns.scatterplot(x=waypoints[:, 1], y=waypoints[:, 0], s=25, alpha=0.2, edgecolor=None)

        figname = f"spatial_coverage.png"
        pt.savefig(os.path.join(self.figure_path, figname))

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

        pt.figure(figsize=graphics.get_standard('fig_size'))
        pt.plot(np.arange(1, len(best_f) + 1), best_f, marker='o')
        pt.xlabel('Generation')
        pt.ylabel('Best Objective Value')
        pt.title('Convergence Plot')
        pt.grid(True)
        pt.savefig(os.path.join(self.figure_path, 'genetic_algorithm_convergence.png'))
        pt.cla()
        pt.close()
