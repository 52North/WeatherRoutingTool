import logging
import math

import numpy as np
from geographiclib.geodesic import Geodesic
from pymoo.core.mutation import Mutation

from WeatherRoutingTool.config import Config
from WeatherRoutingTool.constraints.constraints import ConstraintsList
from WeatherRoutingTool.algorithms.genetic import utils
from WeatherRoutingTool.algorithms.genetic.patcher import PatchFactory, PatcherBase

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
    Base class for Mutation with candidate rejection if mutated routes validate constraints.

    - generate offsprings using sub-class' implementation of ``crossover`` function,
    - rejects offspring that violates constraints based on the config variable ``GENETIC_REPAIR_TYPE``

        - if ``GENETIC_REPAIR_TYPE="no_repair"``, ``constraints_rejection`` is set to ``True`` and offspring that violates
          constraints is rejected such that the parents are returned,
        - if ``GENETIC_REPAIR_TYPE`` is set to any valid repair strategy, ``constraints_rejection`` is set to ``False``
          and all crossover candidates are accepted,
    - counts the number of tried and successful mutations (ignoring mutation probability handled by base class Mutation)


    :param Nof_mutation_tries: Number of initiated mutations.
    :type Nof_mutation_tries: int
    :param Nof_mutation_success: Number of mutations that do not violate constraints.
    :type Nof_mutation_success: int
    :param mutation_type: Name of the mutation type (optional).
    :type mutation_type: str
    :param constraints_list: List of constraints to be validated.
    :type constraints_list: ConstraintsList
    :param constraints_rejection: If ``True``, crossover candidates that violate constraints are rejected. If ``False``,
        all crossover candidates are accepted. The variable is set based on config variable ``GENETIC_REPAIR_TYPE``.
        Defaults to ``True``.
    :type constraints_rejection: bool
    """

    Nof_mutation_tries: int
    Nof_mutation_success: int

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
        self.Nof_mutation_tries = 0
        self.Nof_mutation_success = 0
        self.constraints_rejection = True
        self.config = config

        if not (config.GENETIC_REPAIR_TYPE == ["no_repair"]):
            self.constraints_rejection = False

    def print_mutation_statistics(self):
        logger.info(f'{self.mutation_type} statistics:')
        logger.info('Nof_mutation_tries: ' + str(self.Nof_mutation_tries))
        logger.info('Nof_mutation_success: ' + str(self.Nof_mutation_success))

    def _do(self, problem, X, **kw):
        """
        Implementation of mutation of route matrix with candidate rejection if mutated routes validate constraints.

        :param problem: Routing problem.
        :type: RoutingProblem
        :param X: Route matrix in the form of ``np.array([[route_0], [route_1], ...])`` with
            ``route_i=np.array([[lat_0, lon_0], [lat_1,lon_1], ...])``. X.shape = (n_routes, 1, n_waypoints, 2).
            Access i'th route as ``X[i,0]`` and the j'th coordinate pair off the i'th route as ``X[i,0][j, :]``.
        :type X: np.array
        :return: Mutated route matrix. Same structure as for ``X``.
        :rtype: np.array
        """

        for i, (rt,) in enumerate(X):
            self.Nof_mutation_tries += 1
            mut_temp = self.mutate(problem, rt, **kw)

            if (not utils.get_constraints(mut_temp, self.constraints_list)) or (not self.constraints_rejection):
                self.Nof_mutation_success += 1
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

    def __init__(
            self,
            gcr_dist: float = 1e5,
            n_updates: int = 1,
            plateau_size: int = 3,
            plateau_slope: int = 2,
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

    def random_walk(
            self,
            point: tuple[float, float],
            dist: float = 1e4,
            bearing: float = 45.0,
    ) -> tuple[float, float]:
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

        lat0, lon0 = point
        result = Geodesic.WGS84.Direct(lat0, lon0, bearing, dist)
        lat2 = result["lat2"]
        lon2 = result["lon2"]
        return lat2, lon2

    def mutate(self, problem, rt, **kw):
        """
        Function vor RandomPlateauMutation.

        A set of four waypoints is selected:

        - a plateau center that is chosen on a random basis,
        - two plateau edges which are the waypoints ``self.plateau_size``/2 waypoints before and behind the plateau center,
        - two connectors which are the waypoints ``self.plateau_slope`` before and behind the plateau edges.

        The plateau edges are moved in the same direction to one of their N-4 neighbourhood positions as for random-walk
        mutation. A plateau is drawn by connecting the plateau edges to the connectors and to each other via great circle
        routes.

        Only routes which are long enough are mutated. Routes which are smaller or of size
        ``2 * self.plateau_slope + self.plateau_size`` are returned as they are.

        :param problem: routing problem
        :type: RoutingProblem
        :params rt: route to be mutated
        :type rt: np.array([[lat_0, lon_0], [lat_1,lon_1], ...]),
        :return: mutated route
        :rtype: np.array([[lat_0, lon_0], [lat_1,lon_1], ...]),
        """

        for _ in range(0, self.n_updates):
            plateau_length = 2 * self.plateau_slope + self.plateau_size
            if len(rt[0]) < plateau_length + 1:  # only mutate routes that are long enough
                continue

            # obtain indices for plateau generation
            rindex = np.random.randint(np.ceil(plateau_length / 2), rt.shape[0] - np.ceil(plateau_length / 2))
            i_plateau_start = int(rindex - np.ceil((self.plateau_size - 1) / 2))
            i_plateau_end = int(rindex + np.ceil((self.plateau_size - 1) / 2))
            i_slope_start = int(i_plateau_start - self.plateau_slope)
            i_slope_end = int(i_plateau_end + self.plateau_slope)

            # mutate plateau edges by random walk in same direction
            p1_orig = rt[i_plateau_start]
            p2_orig = rt[i_plateau_end]
            bearing = np.random.choice([45, 135, 225, 315])
            rt[i_plateau_start] = self.random_walk(
                point=p1_orig,
                dist=self.dist,
                bearing=bearing
            )
            rt[i_plateau_end] = self.random_walk(
                point=p2_orig,
                dist=self.dist,
                bearing=bearing
            )

            # obtain subsections, slope & plateau via gcr patching
            dist_one_orig = rt[:i_slope_start]

            if i_slope_start == 0:
                dist_one_orig = [rt[0]]
            dist_one_patched = self.patchfn.patch(
                src=tuple(rt[i_slope_start]),
                dst=tuple(rt[i_plateau_start]),
                npoints=self.plateau_slope,
            )
            dist_plateau_patched = self.patchfn.patch(
                src=tuple(rt[i_plateau_start]),
                dst=tuple(rt[i_plateau_end]),
                npoints=self.plateau_size,
            )
            dist_two_patched = self.patchfn.patch(
                src=tuple(rt[i_plateau_end]),
                dst=tuple(rt[i_slope_end]),
                npoints=self.plateau_slope,
            )
            dist_two_orig = rt[i_slope_end:]

            # combine subsections
            rt = np.concatenate([
                dist_one_orig,
                dist_one_patched[1:],
                dist_plateau_patched[1:],
                dist_two_patched[1:],
                dist_two_orig[1:]
            ])
        return rt


class RouteBlendMutation(MutationConstraintRejection):
    """
    Mutates routes by smoothening with a bezier curve.

    Generates a bezier curve between two randomly selected indices and infills
    it with 2x the number of waypoints previously present in the selected range.
    """

    def __init__(
            self,
            **kw
    ):
        """
            Initialisation function for RouteBlendMutation.

            For definition of kw see description on MutationConstraintRejection.
        """
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
        curve = np.zeros((n_points, 2))

        for i in range(n + 1):
            bernstein = math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
            curve += np.outer(bernstein, control_points[i])

        return curve

    def mutate(self, problem, rt, **kw):

        for _ in range(3):
            p1 = np.random.randint(0, rt.shape[0])
            if rt.shape[0] - p1 <= 3:  # retry
                continue

            p2 = p1 + np.random.randint(3, min(10, rt.shape[0] - p1))
            break

        n_points = (p2 - p1) * 2

        rt = np.concatenate(
            [rt[:p1], self.bezier_curve(rt[p1:p2], n_points), rt[p2:]], axis=0)
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
            point: tuple[float, float],
            dist: float = 1e4,
            bearing: float = 45.0,
    ) -> tuple[float, float]:
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
        lat0, lon0 = point
        result = Geodesic.WGS84.Direct(lat0, lon0, bearing, dist)
        lat2 = result["lat2"]
        lon2 = result["lon2"]
        return lat2, lon2

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


# factory
# ----------
class MutationFactory:
    @staticmethod
    def get_mutation(
            config: Config,
            constraints_list: None
    ) -> Mutation:

        if "no_mutation" in config.GENETIC_MUTATION_TYPE:
            logger.debug('Setting mutation type of genetic algorithm to "no_mutation".')
            return NoMutation()

        if "random" in config.GENETIC_MUTATION_TYPE:
            logger.debug('Setting mutation type of genetic algorithm to "random".')
            return RandomMutationsOrchestrator(
                opts=[
                    RandomPlateauMutation(config=config, constraints_list=constraints_list),
                    RouteBlendMutation(config=config, constraints_list=constraints_list)
                ], )

        if "rndm_walk" in config.GENETIC_MUTATION_TYPE:
            logger.debug('Setting mutation type of genetic algorithm to "random_walk".')
            return RandomWalkMutation(config=config, constraints_list=constraints_list)

        if "rndm_plateau" in config.GENETIC_MUTATION_TYPE:
            logger.debug('Setting mutation type of genetic algorithm to "random_plateau".')
            return RandomPlateauMutation(config=config, constraints_list=constraints_list)

        if "route_blend" in config.GENETIC_MUTATION_TYPE:
            logger.debug('Setting mutation type of genetic algorithm to "route_blend".')
            return RouteBlendMutation(config=config, constraints_list=constraints_list)

        raise NotImplementedError(f'The mutation type {config.GENETIC_MUTATION_TYPE} is not implemented.')
