from pymoo.core.mutation import Mutation

from pyproj import Geod
import numpy as np

import logging
import math

from WeatherRoutingTool.config import Config

logger = logging.getLogger("WRT.genetic.mutation")


# base class
# ----------
class MutationBase(Mutation):
    """Base Mutation Class

    Kept for consistency with how crossover is implemented and to provide
    super-class level orchestration of Mutation implementations.
    """

    def _do(self, problem, X, **kw):
        return self.mutate(problem, X, **kw)

    def mutate(self, problem, X, **kw):
        return X


# mutation variants
# ----------
class NoMutation(MutationBase):
    def mutate(self, problem, X, **kw):
        return super().__init__()


class RandomWalkMutation(Mutation):
    def __init__(
            self,
            ellps: str = "WGS84",
            dist: int = int(1e4),
            n_updates: int = 10,
            **kw
    ):
        super().__init__(**kw)

        self.geod = Geod(ellps=ellps)
        self.dist = dist
        self.n_updates = n_updates

    def random_walk(
            self,
            point,
            dist = int(1e4),
            bearing: int = 45
    ) -> tuple[float, float]:
        """Randomly pick an N4 neighbour of a waypoint"""

        x, y = point

        lat, lon, back_azimuth = self.geod.fwd(x, y, bearing, dist)
        return lat, lon

    def mutate(self, problem, X, **kw):
        for i, (rt,) in enumerate(X):
            for _ in range(self.n_updates):
                rindex = np.random.randint(1, rt.shape[0] - 1)
                p1 = rt[rindex]

                p2 = self.random_walk(
                    point=p1,
                    dist=self.dist,
                    bearing=np.random.choice([45, 135, 225, 315]), )
                rt[rindex] = p2
        return X


class RouteBlendMutation(MutationBase):
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
            bernstein = math.comb(n, i) * (t**i) * ((1 - t) ** (n - i))
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


# factory
# ----------
class MutationFactory:
    @staticmethod
    def get_mutation(config: Config) -> Mutation:
        return RandomWalkMutation()
