import logging
import math

import numpy as np
from geographiclib.geodesic import Geodesic
from pymoo.core.mutation import Mutation

from WeatherRoutingTool.config import Config
from WeatherRoutingTool.algorithms.genetic.patcher import PatchFactory

logger = logging.getLogger("WRT.genetic.mutation")


# base class
# ----------
class MutationBase(Mutation):
    """Base Mutation Class

    Kept for consistency and to provide super-class level management of
    Mutation implementations.
    """

    def _do(self, problem, X, **kw):
        return self.mutate(problem, X, **kw)

    def mutate(self, problem, X, **kw):
        raise NotImplementedError('No mutation method is implemented for the base class.')


# mutation variants
# ----------
class NoMutation(MutationBase):
    """Empty Mutation class for testing"""

    def mutate(self, problem, X, **kw):
        return X


class RandomWalkMutation(MutationBase):
    """Moves a random waypoint in an individual in the direction of a random bearing
    by a distance specified by the `gcr_dist` parameter.
    The entire task is repeated `n_updates` number of times to produce substantial variation.

    :param gcr_dist: The distance by which to move the point
    :type gcr_dist: float
    :param n_updates: Number of iterations to repeat the random walk operation
    :type n_updates: int
    """

    def __init__(
            self,
            config: Config,
            gcr_dist: float = 1e5,
            n_updates: int = 1,
            plateau_size: int = 3,
            plateau_slope: int = 2,
            **kw
    ):
        super().__init__(**kw)

        if plateau_size % 2 != 1:
            raise ValueError('The plateau_size of RandomWalkMutation needs to be an uneven number!')

        self.geod = Geodesic.WGS84
        self.dist = gcr_dist
        self.n_updates = n_updates
        self.config = config
        self.plateau_size = plateau_size  # uneven
        self.plateau_slope = plateau_slope

    def random_walk(
            self,
            point: tuple[float, float],
            dist: float = 1e4,
            bearing: float = 45.0,
    ) -> tuple[float, float]:
        """Pick an N4 neighbour of a waypoint

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
        result = self.geod.Direct(lat0, lon0, bearing, dist)
        lat2 = result["lat2"]
        lon2 = result["lon2"]
        return lat2, lon2

    def mutate(self, problem, X, **kw):
        '''
        Function vor RandomPlateauMutation.

        :params X: route matrix
        :type X: np.array([[route_0], [route_1], ...]) with route_i=np.array([[lat_0, lon_0], [lat_1,lon_1], ...]),
                 X.shape = (n_routes, 1, n_waypoints, 2)
                 access i'th route as X[i,0] and the j'th coordinate pair off the i'th route as X[i,0][j, :]
        '''

        patchfn = PatchFactory.get_patcher(patch_type="gcr", config=self.config, application="Random walk mutation")

        for i, (rt,) in enumerate(X):
            print('i: ', i)
            for _ in range(0, self.n_updates):
                print('rt: ', rt)
                plateau_length = 2 * self.plateau_slope + self.plateau_size
                if len(rt[0]) < plateau_length + 1:  # only mutate routes that are long enough
                    continue

                # obtain indices for plateau generation
                rindex = np.random.randint(np.ceil(plateau_length / 2), rt.shape[0] - np.ceil(plateau_length / 2))
                i_plateau_start = int(rindex - np.ceil((self.plateau_size - 1) / 2))
                i_plateau_end = int(rindex + np.ceil((self.plateau_size - 1) / 2))
                i_slope_start = int(i_plateau_start - self.plateau_slope)
                i_slope_end = int(i_plateau_end + self.plateau_slope)

                print('index: ', rindex)
                print('plateau slope start: ', i_slope_start)
                print('plateau start: ', i_plateau_start)
                print('plateau end: ', i_plateau_end)
                print('plateau slope end: ', i_slope_end)

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
                print('mutated p1: ', rt[i_plateau_start])
                print('mutated p2: ', rt[i_plateau_end])

                # obtain subsections, slope & plateau via gcr patching
                dist_one_orig = rt[:i_slope_start]
                print('shape_dist_one_orig: ', dist_one_orig.shape)
                if i_slope_start == 0:
                    dist_one_orig = [rt[0]]
                dist_one_patched = patchfn.patch(
                    src=tuple(rt[i_slope_start]),
                    dst=tuple(rt[i_plateau_start]),
                    npoints=self.plateau_slope,
                )
                dist_plateau_patched = patchfn.patch(
                    src=tuple(rt[i_plateau_start]),
                    dst=tuple(rt[i_plateau_end]),
                    npoints=self.plateau_size,
                )
                dist_two_patched = patchfn.patch(
                    src=tuple(rt[i_plateau_end]),
                    dst=tuple(rt[i_slope_end]),
                    npoints=self.plateau_slope,
                )
                dist_two_orig = rt[i_slope_end:]

                # combine subsections
                print('dist_one_orig: ', dist_one_orig)
                print('dist_one_patched: ', dist_one_patched)
                print('dist_plateau: ', dist_plateau_patched)
                print('dist_two_patched: ', dist_two_patched)
                print('dist_two_orig: ', dist_two_orig)
                Y = np.concatenate([
                    dist_one_orig,
                    dist_one_patched[1:],
                    dist_plateau_patched[1:],
                    dist_two_patched[1:],
                    dist_two_orig[1:]
                ])
                print('X: ', X.shape)
                print('X[i,0]: ', X[i, 0])
                print('Y: ', Y.shape)
                print('Y: ', Y)
                X[i, 0] = Y

        return X


class RouteBlendMutation(MutationBase):
    """Generates a bezier curve between two randomly selected indices and infills
    it with 2x the number of waypoints previously present in the selected range.
    """

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

    def mutate(self, problem, X, **kw):
        for i, (rt,) in enumerate(X):
            for _ in range(3):
                p1 = np.random.randint(0, rt.shape[0])
                if rt.shape[0] - p1 <= 3:  # retry
                    continue

                p2 = p1 + np.random.randint(3, min(10, rt.shape[0] - p1))
                break
            else:
                continue

            n_points = (p2 - p1) * 2

            X[i, 0] = np.concatenate(
                [rt[:p1], self.bezier_curve(rt[p1:p2], n_points), rt[p2:]], axis=0)
        return X


# ----------
class RandomMutationsOrchestrator(MutationBase):
    """Select a mutation operator at random and apply it over the population

    :param opts: List of Mutation classes
    :type opts: list[Mutation]
    """

    def __init__(self, opts, **kw):
        super().__init__(**kw)

        self.opts = opts

    def _do(self, problem, X, **kw):
        opt = self.opts[np.random.randint(0, len(self.opts))]
        return opt._do(problem, X, **kw)


# factory
# ----------
class MutationFactory:
    @staticmethod
    def get_mutation(config: Config) -> Mutation:

        if "no_mutation" in config.GENETIC_MUTATION_TYPE:
            logger.debug('Setting mutation type of genetic algorithm to "no_mutation".')
            return NoMutation()

        if "random" in config.GENETIC_MUTATION_TYPE:
            logger.debug('Setting mutation type of genetic algorithm to "random".')
            return RandomMutationsOrchestrator(
                opts=[
                    RandomWalkMutation(),
                    RouteBlendMutation()
                ], )

        if "rndm_walk" in config.GENETIC_MUTATION_TYPE:
            logger.debug('Setting mutation type of genetic algorithm to "random_walk".')
            return RandomWalkMutation()

        if "route_blend" in config.GENETIC_MUTATION_TYPE:
            logger.debug('Setting mutation type of genetic algorithm to "route_blend".')
            return RouteBlendMutation()

        raise NotImplementedError(f'The mutation type {config.GENETIC_MUTATION_TYPE} is not implemented.')
