import logging
import math
import os
import random
from operator import add, sub

import numpy as np
from geographiclib.geodesic import Geodesic
from pymoo.core.mutation import Mutation

import WeatherRoutingTool.utils.graphics as graphics
from WeatherRoutingTool.config import Config
from WeatherRoutingTool.constraints.constraints import ConstraintsList
from WeatherRoutingTool.algorithms.genetic import utils
from WeatherRoutingTool.algorithms.genetic.patcher import PatchFactory, PatcherBase
from WeatherRoutingTool.utils.maps import Map

logger = logging.getLogger("WRT.genetic.mutation")


class MutationBase(Mutation):
    """Base Mutation Class

    Kept for consistency and to provide super-class level management of
    Mutation implementations.
    """

    def __init__(self, prob: float = 1.0, prob_var: float = None):
        super().__init__(prob, prob_var)

    def _do(self, problem, X, **kw):
        raise NotImplementedError('No mutation method is implemented for MutationBase.')


class MutationConstraintRejection(Mutation):
    """
    Base class for Mutation with candidate rejection if mutated routes violate constraints.

    - generate offsprings using sub-class' implementation of ``mutation`` function,
    - rejects offspring that violates constraints based on the config variable ``GENETIC_REPAIR_TYPE``

        - if ``GENETIC_REPAIR_TYPE="no_repair"``, ``constraints_rejection`` is set to ``True`` and offspring that
        violates constraints is rejected such that the parents are returned,
        - if ``GENETIC_REPAIR_TYPE`` is set to any valid repair strategy, ``constraints_rejection`` is set to ``False``
          and all mutated candidates are accepted,
    - counts the number of tried and successful mutations (ignoring mutation probability handled by base class Mutation)


    :param nof_mutation_tries: Number of initiated mutations.
    :type nof_mutation_tries: int
    :param nof_mutation_success: Number of mutations that do not violate constraints.
    :type nof_mutation_success: int
    :param mutation_type: Name of the mutation type (optional).
    :type mutation_type: str
    :param constraints_list: List of constraints to be validated.
    :type constraints_list: ConstraintsList
    :param constraints_rejection: If ``True``, mutated candidates that violate constraints are rejected. If ``False``,
        all mutated candidates are accepted. The variable is set based on config variable ``GENETIC_REPAIR_TYPE``.
        Defaults to ``True``.
    :type constraints_rejection: bool
    """

    nof_mutation_tries: int
    nof_mutation_success: int

    mutation_type: str

    constraints_list: ConstraintsList
    constraints_rejection: bool

    def __init__(
            self,
            mutation_type: str,
            config: Config,
            constraints_list: ConstraintsList,
            prob: float = 1.0,
            prob_var: float = None
    ):
        """
            Initialisation function for MutationConstraintRejection.

            :param mutation_type: Name of mutation strategy.
            :type mutation_type: str
            :param config: Config object.
            :type config: Config
            :param constraints_list: List of constraint objects.
            :type constraints_list: ConstraintsList
            :param prob: Mutation probability.
            :type prob: Real(BoundedVariable)
            :param prob_var: Range of mutation probability.
            :type prob: Real(BoundedVariable)
        """
        super().__init__(prob=prob, prob_var=prob_var)

        self.constraints_list = constraints_list
        self.mutation_type = mutation_type
        self.nof_mutation_tries = 0
        self.nof_mutation_success = 0
        self.constraints_rejection = True
        self.config = config

        if not (config.GENETIC_REPAIR_TYPE == ["no_repair"]):
            self.constraints_rejection = False

    def print_mutation_statistics(self):
        logger.info(f'{self.mutation_type} statistics:')
        logger.info('nof_mutation_tries: ' + str(self.nof_mutation_tries))
        logger.info('nof_mutation_success: ' + str(self.nof_mutation_success))

    def _do(self, problem, X, **kw):
        """
        Implementation of mutation of route matrix with candidate rejection if mutated routes validate constraints.

        :param problem: Routing problem.
        :type: RoutingProblem
        :param X: Route matrix in the form of ``np.array([[route_0], [route_1], ...])`` with
            ``route_i=np.array([[lat_0, lon_0, v_0], [lat_1,lon_1, v_1], ...])``.
            X.shape = (n_routes, 1, n_waypoints, 3).
            Access i'th route as ``X[i,0]`` and the j'th coordinate pair off the i'th route as ``X[i,0][j, :]``.
        :type X: np.array
        :return: Mutated route matrix. Same structure as for ``X``.
        :rtype: np.array
        """

        for i, (rt,) in enumerate(X):
            self.nof_mutation_tries += 1
            mut_temp = self.mutate(problem, rt, **kw)

            if (not utils.get_constraints(mut_temp, self.constraints_list)) or (not self.constraints_rejection):
                self.nof_mutation_success += 1
                X[i, 0] = mut_temp

        return X

    def mutate(self, problem, rt, **kw):
        raise NotImplementedError('No mutation method is implemented for MutationConstraintRejection.')


# mutation variants
# ----------
class NoMutation(MutationBase):
    """Empty Mutation class for testing."""

    def _do(self, problem, X, **kw):
        return X


class RandomPlateauMutation(MutationConstraintRejection):
    """
    Mutates Routes by adding a 'plateau'.

    Selects a waypoint on a random basis and calls it the 'plateau center'. The route is tilted around this plateau
    center such that the shape resembles a plateau.

    :param Config: Config object.
    :type Config: Config
    :param dist: Distance by which the plateau edges are mutated.
    :type dist: float
    :param n_updates: Number of plateaus that are introduced.
    :type n_updates: int
    :param plateau_size: Number of waypoints that form the top of the plateau.
    :type plateau_size: int
    :param plateau_slope: Number of waypoints that form the side of the plateau.
    :type plateau_slope: int
    :param patchnf: Gcr patcher for connecting plateau edges and connectors.
    :type patchnf: PatcherBase
    """

    config: Config
    dist: float
    n_updates: int
    plateau_size: int
    plateau_slope: int
    patchnf: PatcherBase
    route_count: int

    def __init__(
            self,
            gcr_dist: float = 1e4,
            n_updates: int = 1,
            plateau_size: int = 3,
            plateau_slope: int = 3,
            **kw
    ):
        """
            Initialisation function for RandomPlateauMutation.

            For definition of kw see description on MutationConstraintRejection.

            :param dist: Distance by which the plateau edges are mutated.
            :type dist: float
            :param n_updates: Number of plateaus that are introduced.
            :type n_updates: int
            :param plateau_size: Number of waypoints that form the top of the plateau.
            :type plateau_size: int
            :param plateau_slope: Number of waypoints that form the side of the plateau.
            :type plateau_slope: int
        """
        super().__init__(
            mutation_type="RandomPlateauMutation",
            **kw
        )

        if plateau_size % 2 != 1:
            raise ValueError('The plateau_size of RandomPlateauMutation needs to be an uneven number!')

        self.n_updates = n_updates
        self.plateau_size = plateau_size  # uneven
        self.plateau_slope = plateau_slope

        self.dist = gcr_dist
        self.patchfn = PatchFactory.get_patcher(patch_type="gcr", config=self.config,
                                                application="Route plateau mutation")
        self.route_count = 0

    def random_walk(
            self,
            point: tuple[float, float, float],
            dist: float = 1e4,
            bearing: float = 45.0,
    ) -> tuple[float, float, float]:
        """Pick an N4 neighbour of a waypoint.

        :param point: (lat, lon) in degrees.
        :type point: tuple[float, float]
        :param dist: distance in meters
        :type dist: float
        :param bearing: Azimuth in degrees (clockwise from North)
        :type bearing: float
        :return: (lat, lon) in degrees.
        :rtype: tuple[float, float]
        """

        lat0, lon0, speed = point
        result = Geodesic.WGS84.Direct(lat0, lon0, bearing, dist)
        lat2 = result["lat2"]
        lon2 = result["lon2"]
        return lat2, lon2, speed

    def mutate(self, problem, rt, **kw):
        """
        Function vor RandomPlateauMutation.

        A set of four waypoints is selected:

        - a plateau center that is chosen on a random basis,
        - two plateau edges which are the waypoints ``self.plateau_size``/2 waypoints before and behind the plateau
          center,
        - two connectors which are the waypoints ``self.plateau_slope`` before and behind the plateau edges.

        The plateau edges are moved in the same direction to one of their N-4 neighbourhood positions as for random-walk
        mutation. A plateau is drawn by connecting the plateau edges to the connectors and to each other via great
        circle routes.

        Only routes which are long enough are mutated. Routes which are smaller or of size
        ``2 * self.plateau_slope + self.plateau_size`` are returned as they are.

        :param problem: routing problem
        :type: RoutingProblem
        :params rt: route to be mutated
        :type rt: np.array([[lat_0, lon_0, v_0], [lat_1,lon_1, v_1], ...]),
        :return: mutated route
        :rtype: np.array([[lat_0, lon_0, v_0], [lat_1,lon_1, v_1], ...]),
        """
        debug = False

        # test whether input route rt has the correct shape
        assert len(rt.shape) == 2
        assert rt.shape[1] == 3
        route_length = rt.shape[0]
        plateau_length = 2 * self.plateau_slope + self.plateau_size - 2
        rt_new = np.full(rt.shape, -99.)

        if route_length <= plateau_length + 1:  # only mutate routes that are long enough
            return rt

        if debug:
            print('################################')
            print('original rt: ', rt)

        for _ in range(0, self.n_updates):
            # obtain indices for plateau generation
            rindex = np.random.randint(np.ceil(plateau_length / 2), route_length - np.ceil(plateau_length / 2))
            i_plateau_start = int(rindex - np.ceil((self.plateau_size - 1) / 2))
            i_plateau_end = int(rindex + np.ceil((self.plateau_size - 1) / 2))
            i_slope_start = int(i_plateau_start - self.plateau_slope) + 1
            i_slope_end = int(i_plateau_end + self.plateau_slope) - 1

            if debug:
                print('Indices: ')
                print('     plateau mid: ', rindex)
                print('     i_slope_start: ', i_slope_start)
                print('     i_plateau_start: ', i_plateau_start)
                print('     i_plateau_end: ', i_plateau_end)
                print('     i_slope_end: ', i_slope_end)

            # mutate plateau edges by random walk in same direction
            p1_orig = rt[i_plateau_start]
            p2_orig = rt[i_plateau_end]
            bearing = np.random.randint(0, 360)
            rt_new[i_plateau_start] = self.random_walk(
                point=p1_orig,
                dist=self.dist,
                bearing=bearing
            )
            rt_new[i_plateau_end] = self.random_walk(
                point=p2_orig,
                dist=self.dist,
                bearing=bearing
            )
            if debug:
                print('Mutated plateau edges:')
                print('     point 1: ', rt_new[i_plateau_start])
                print('     point 2: ', rt_new[i_plateau_end])

            # obtain subsections, slope & plateau via gcr patching
            dist_one_orig = rt[:i_slope_start]
            if i_slope_start == 0:
                dist_one_orig = [rt[0]]
            dist_one_patched_full = self.patchfn.patch(
                src=tuple(rt[i_slope_start]),
                dst=tuple(rt_new[i_plateau_start]),
                npoints=self.plateau_slope - 1,
            )
            dist_one_patched = dist_one_patched_full[:-1]
            dist_plateau_patched = self.patchfn.patch(
                src=tuple(rt_new[i_plateau_start]),
                dst=tuple(rt_new[i_plateau_end]),
                npoints=self.plateau_size - 1,
            )
            dist_two_patched_full = self.patchfn.patch(
                src=tuple(rt_new[i_plateau_end]),
                dst=tuple(rt[i_slope_end]),
                npoints=self.plateau_slope - 1,
            )
            dist_two_patched = dist_two_patched_full[1:]
            dist_two_orig = rt[i_slope_end + 1:]

            if debug:
                print('Full segments: ')
                print('     dist_on_orig: ', dist_one_orig)
                print('     dist_one_patched (full): ', dist_one_patched_full)
                print('     dist_plateau_patched: ', dist_plateau_patched)
                print('     dist_two_patched: (full)', dist_two_patched_full)
                print('     dist_two_orig: ', dist_two_orig)

                print('Cut segments: ')
                print('     dist_one_patched: ', dist_one_patched)
                print('     dist_two_patched: ', dist_two_patched)

            # combine subsections
            rt_new = np.concatenate([
                dist_one_orig,
                dist_one_patched,
                dist_plateau_patched,
                dist_two_patched,
                dist_two_orig
            ])

        if debug:
            import cartopy.crs as ccrs
            import matplotlib.pyplot as plt
            print('mutated rt: ', rt_new)
            map = Map(rt[0][0], rt[0][1], rt[-1][0], rt[-1][1])
            input_crs = ccrs.PlateCarree()

            fig, ax = graphics.generate_basemap(
                map=map.get_var_tuple(),
                depth=None,
                start=rt[0],
                finish=rt[-1],
                show_depth=False
            )
            ax.plot(rt[:, 1], rt[:, 0], color="firebrick", transform=input_crs, marker="o")
            ax.plot(rt_new[:, 1], rt_new[:, 0], color="blue", transform=input_crs, marker="o")
            figname = 'mutated_route' + str(self.route_count)
            figurepath = graphics.get_figure_path()
            plt.savefig(os.path.join(figurepath, figname))
            self.route_count += 1

        return rt_new


class RouteBlendMutation(MutationConstraintRejection):
    """
    Mutates routes by smoothening with a bezier curve.

    Generates a bezier curve between two randomly selected indices and infills
    it with the number of waypoints previously present in the selected range.
    """

    max_lengh: int
    min_length: int

    def __init__(
            self,
            **kw
    ):
        """
            Initialisation function for RouteBlendMutation.

            For definition of kw see description on MutationConstraintRejection.
        """

        self.min_length = 3
        self.max_length = 10
        super().__init__(
            mutation_type="RouteBlendMutation",
            **kw
        )

    @staticmethod
    def bezier_curve(control_points, n_points=100):
        """Generate a Bezier curve given control points.

        :param control_points: List of (x, y) control points
        :type control_points: np.ndarray
        :param n_points: Number of points on the curve
        :type n_points: int
        :return: Array of shape (n_points, 2) with curve coordinates
        :rtype: np.ndarray
        """
        control_points = np.array(control_points)
        n = len(control_points) - 1  # degree
        t = np.linspace(0, 1, n_points)
        curve = np.zeros((n_points, 3))

        for i in range(n + 1):
            bernstein = math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
            curve += np.outer(bernstein, control_points[i])

        return curve

    def mutate(self, problem, rt, **kw):
        # test shape of input route
        assert len(rt.shape) == 2
        assert rt.shape[1] == 3
        route_length = rt.shape[0]

        # only mutate routes that are long enough
        if route_length < self.min_length:
            return rt

        start = np.random.randint(0, route_length - self.min_length)
        length = np.random.randint(self.min_length, min(self.max_length, route_length - start))
        end = start + length
        n_points = length

        rt = np.concatenate([rt[:start], self.bezier_curve(rt[start:end], n_points), rt[end:]], axis=0)

        return rt


class RandomWalkMutation(MutationConstraintRejection):
    """
    Mutates routes by moving single waypoints on a random basis.

    Moves a random waypoint in an individual in the direction of a random bearing
    by a distance specified by the `gcr_dist` parameter.
    The entire task is repeated `n_updates` number of times to produce substantial variation.

    :param gcr_dist: The distance by which to move the point
    :type gcr_dist: float
    :param n_updates: Number of iterations to repeat the random walk operation
    :type n_updates: int
    """
    dist: float
    n_updates: int

    def __init__(
            self,
            gcr_dist: float = 1e4,
            n_updates: int = 10,
            **kw
    ):
        """
            Initialisation function for RouteBlendMutation.

            For definition of kw see description on MutationConstraintRejection.

            :param gcr_dist: The distance by which to move the point
            :type gcr_dist: float
            :param n_updates: Number of iterations to repeat the random walk operation
            :type n_updates: int
        """
        super().__init__(
            mutation_type="RandomWalkMutation",
            **kw
        )

        self.dist = gcr_dist
        self.n_updates = n_updates

    def random_walk(
            self,
            point: tuple[float, float, float],
            dist: float = 1e4,
            bearing: float = 45.0,
    ) -> tuple[float, float, float]:
        """Pick an N4 neighbour of a waypoint.

        :param point: (lat, lon) in degrees.
        :type point: tuple[float, float, float]
        :param dist: distance in meters
        :type dist: float
        :param bearing: Azimuth in degrees (clockwise from North)
        :type bearing: float
        :return: (lat, lon) in degrees.
        :rtype: tuple[float, float, float]
        """
        lat0, lon0, speed = point
        result = Geodesic.WGS84.Direct(lat0, lon0, bearing, dist)
        lat2 = result["lat2"]
        lon2 = result["lon2"]
        return lat2, lon2, speed

    def mutate(self, problem, rt, **kw):
        for _ in range(self.n_updates):
            rindex = np.random.randint(1, rt.shape[0] - 1)
            p1 = rt[rindex]

            p2 = self.random_walk(
                point=p1,
                dist=self.dist,
                bearing=np.random.choice([45, 135, 225, 315]), )
            rt[rindex] = p2
        return rt


# ----------
class RandomMutationsOrchestrator(MutationBase):
    """Select a mutation operator at random and apply it to the population.

    :param opts: List of Mutation classes.
    :type opts: list[Mutation]
    """

    def __init__(self, opts, **kw):
        super().__init__(**kw)

        self.opts = opts

    def _do(self, problem, X, **kw):
        opt = self.opts[np.random.randint(0, len(self.opts))]
        return opt._do(problem, X, **kw)

    def print_mutation_statistics(self):
        for opt in self.opts:
            opt.print_mutation_statistics()


class RandomPercentageChangeSpeedMutation(MutationConstraintRejection):
    """
    Ship speed mutation class.
    Speed values are mutated by randomly adding or subtracting a percentage. The percentage is randomly chosen
    between 0 and a fixed maximum percentage (20 %).
    """
    n_updates: int
    config: Config

    def __init__(self, n_updates: int = 10, **kw):
        super().__init__(
            mutation_type="RandomPercentageChangeSpeedMutation",
            **kw
        )
        self.n_updates = n_updates
        self.change_percent_max = 0.2

    def mutate(self, problem, rt, **kw):
        try:
            indices = random.sample(range(0, rt.shape[0] - 1), self.n_updates)
        except ValueError:
            indices = range(0, rt.shape[0] - 1)
        ops = (add, sub)
        for i in indices:
            op = random.choice(ops)
            change_percent = random.uniform(0.0, self.change_percent_max)
            new = op(rt[i][2], change_percent * rt[i][2])
            if new < 0:
                new = 0
            elif new > self.config.BOAT_SPEED_MAX:
                new = self.config.BOAT_SPEED_MAX
            rt[i][2] = new
        return rt


class GaussianSpeedMutation(MutationConstraintRejection):
    """
    Ship speed mutation class.
    Speed values are updated by drawing random samples from a Gaussian distribution. The mean value of the distribution
    is half of the maximum boat speed. The standard deviation is 1/6 of the maximum boat speed.
    """
    n_updates: int
    config: Config

    def __init__(self, n_updates: int = 10, **kw):
        super().__init__(
            mutation_type="GaussianSpeedMutation",
            **kw
        )
        self.n_updates = n_updates
        # FIXME: these numbers should be carefully evaluated
        # ~99.7 % in interval (0, BOAT_SPEED_MAX)
        self.mu = 0.5 * self.config.BOAT_SPEED_MAX
        self.sigma = self.config.BOAT_SPEED_MAX / 6

    def mutate(self, problem, rt, **kw):
        try:
            indices = random.sample(range(0, rt.shape[0] - 1), self.n_updates)
        except ValueError:
            indices = range(0, rt.shape[0] - 1)
        for i in indices:
            new = random.normalvariate(self.mu, self.sigma)
            if new < 0:
                new = 0
            elif new > self.config.BOAT_SPEED_MAX:
                new = self.config.BOAT_SPEED_MAX
            rt[i][2] = new
        return rt


# factory
# ----------
class MutationFactory:
    @staticmethod
    def get_mutation(
            config: Config,
            constraints_list: None
    ) -> Mutation:

        if config.GENETIC_MUTATION_TYPE == "no_mutation":
            logger.debug('Setting mutation type of genetic algorithm to "no_mutation".')
            return NoMutation()

        if config.GENETIC_MUTATION_TYPE == "random":
            logger.debug('Setting mutation type of genetic algorithm to "random".')
            return RandomMutationsOrchestrator(
                opts=[
                    RandomPlateauMutation(config=config, constraints_list=constraints_list),
                    RouteBlendMutation(config=config, constraints_list=constraints_list)
                ], )

        if config.GENETIC_MUTATION_TYPE == "rndm_walk":
            logger.debug('Setting mutation type of genetic algorithm to "random_walk".')
            return RandomWalkMutation(config=config, constraints_list=constraints_list)

        if config.GENETIC_MUTATION_TYPE == "rndm_plateau":
            logger.debug('Setting mutation type of genetic algorithm to "random_plateau".')
            return RandomPlateauMutation(config=config, constraints_list=constraints_list)

        if config.GENETIC_MUTATION_TYPE == "route_blend":
            logger.debug('Setting mutation type of genetic algorithm to "route_blend".')
            return RouteBlendMutation(config=config, constraints_list=constraints_list)

        if config.GENETIC_MUTATION_TYPE == "percentage_change_speed":
            logger.debug('Setting mutation type of genetic algorithm to "percentage_change_speed".')
            return RandomPercentageChangeSpeedMutation(config=config, constraints_list=constraints_list)

        if config.GENETIC_MUTATION_TYPE == "gaussian_speed":
            logger.debug('Setting mutation type of genetic algorithm to "gaussian_speed".')
            return GaussianSpeedMutation(config=config, constraints_list=constraints_list)

        raise NotImplementedError(f'The mutation type {config.GENETIC_MUTATION_TYPE} is not implemented.')
