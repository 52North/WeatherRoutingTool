from pymoo.core.crossover import Crossover

import numpy as np

from datetime import datetime
import logging

from WeatherRoutingTool.constraints.constraints import ConstraintsList
from WeatherRoutingTool.algorithms.genetic import utils
from WeatherRoutingTool.config import Config

logger = logging.getLogger("WRT.genetic.crossover")


# base classes
# ----------
class OffspringRejectionCrossover(Crossover):
    """Offspring—Rejection Crossover Base Class

    Algorithm
    =========

    - Generate offsprings using sub-class' implementation of the `crossover` function
    - Validate if offsprings violate discrete constraints
        - if True, get rid of both offsprings, and return parents
        - if False, return offsprings
    """

    def __init__(
        self,
        departure_time: datetime,
        constraints_list: ConstraintsList,
        prob=.5,
    ):
        super().__init__(n_parents=2, n_offsprings=2, prob=prob)

        self.departure_time = departure_time
        self.constraints_list = constraints_list

    def _do(self, problem, X, **kw):
        # n_parents assumed to be 2
        # n_var assumed to be 1 -> expands into (N, 2)
        n_parents, n_matings, n_var = X.shape

        Y = np.full_like(X, None, dtype=object)

        for k in range(n_matings):
            p1 = X[0, k, 0]
            p2 = X[1, k, 0]

            o1, o2 = self.crossover(p1.copy(), p2.copy())

            if (
                self.route_constraint_violations(o1).any() or
                self.route_constraint_violations(o2).any()
            ):
                Y[0, k, 0] = p1
                Y[1, k, 0] = p2
            else:
                Y[0, k, 0] = o1
                Y[1, k, 0] = o2
        return Y

    def crossover(
        self,
        p1: np.ndarray,
        p2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sub-class' implementation of the crossover function"""

        return p1, p2

    def route_constraint_violations(self, route: np.ndarray) -> np.ndarray:
        """Check if route breaks any discrete constraints

        :param route: list of waypoints
        :dtype route: np.ndarray
        :return: Boolean array of constraint violations per waypoint
        :rtype: np.ndarray
        """

        is_constrained = self.constraints_list.safe_crossing_discrete(
            route[:-1, 0], route[:-1, 1], route[1:, 0], route[1:, 1],
            current_time=self.departure_time,
            is_constrained=[False] * (route.shape[0] - 1), )

        return np.array(is_constrained)


# crossover implementations
# ----------
class SinglePointCrossover(OffspringRejectionCrossover):
    """Single-point crossover operator with great-circle patching"""

    def crossover(self, p1, p2):
        xp1 = np.random.randint(1, p1.shape[0] - 1)
        xp2 = np.random.randint(1, p2.shape[0] - 1)

        gcr_dist = 1e6

        r1 = np.concatenate([
            p1[:xp1],
            utils.great_circle_route(tuple(p1[xp1-1]), tuple(p2[xp2]), distance=gcr_dist),
            p2[xp2:]])

        r2 = np.concatenate([
            p2[:xp2],
            utils.great_circle_route(tuple(p2[xp2-1]), tuple(p1[xp1]), distance=gcr_dist),
            p1[xp1:]])

        return r1, r2


class EdgeRecombinationCrossover(OffspringRejectionCrossover):
    """Two-point crossover operator with great-circle patching"""

    def crossover(self, p1, p2):
        x11 = np.random.randint(1, p1.shape[0] - 2)
        x12 = np.random.randint(x11 + 1, p1.shape[0] - 1)

        x21 = np.random.randint(1, p2.shape[0] - 2)
        x22 = np.random.randint(x21 + 1, p2.shape[0] - 1)

        gcr_dist = 1e6

        r1 = np.concatenate([
            p1[:x11],
            utils.great_circle_route(
                tuple(p1[x11 - 1]), tuple(p2[x21]), distance=gcr_dist, ),
            p2[x21:x22],
            utils.great_circle_route(
                tuple(p2[x22 - 1]), tuple(p1[x12]), distance=gcr_dist, ),
            p1[x12:]
        ])

        r2 = np.concatenate([
            p2[:x21],
            utils.great_circle_route(
                tuple(p2[x21 - 1]), tuple(p1[x11]), distance=gcr_dist, ),
            p1[x11:x12],
            utils.great_circle_route(
                tuple(p1[x12 - 1]), tuple(p2[x22]), distance=gcr_dist, ),
            p2[x22:]
        ])

        return r1, r2


# ----------
class CrossoverFactory:
    @staticmethod
    def get_crossover(config: Config, constraints_list: ConstraintsList):
        # inputs
        departure_time = config.DEPARTURE_TIME

        return EdgeRecombinationCrossover(
            departure_time=departure_time,
            constraints_list=constraints_list,
            prob=.8, )

        return SinglePointCrossover(
            departure_time=departure_time,
            constraints_list=constraints_list,
            prob=.5, )

        # return PMX(
        #     departure_time=departure_time,
        #     constraints_list=constraints_list,
        #     prob=.5, )

        return OffspringRejectionCrossover(
            departure_time=departure_time,
            constraints_list=constraints_list,
            prob=.5, )
