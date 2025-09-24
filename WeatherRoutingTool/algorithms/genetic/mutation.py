from pymoo.core.mutation import Mutation

import numpy as np

from copy import deepcopy
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
class MutationFactory:
    @staticmethod
    def get_mutation(config: Config) -> Mutation:
        return RouteBlendMutation()
