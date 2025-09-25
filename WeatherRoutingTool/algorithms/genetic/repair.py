from pymoo.core.repair import Repair, NoRepair

import numpy as np
import logging

from WeatherRoutingTool.config import Config
from WeatherRoutingTool.constraints.constraints import ConstraintsList
from WeatherRoutingTool.algorithms.genetic import utils, patcher

logger = logging.getLogger("WRT.genetic.repair")


# ----------
class RepairBase(Repair):
    def __init__(self, config: Config):
        super().__init__()

        self.config = config

    def _do(self, problem, X, **kw):
        return self.repairfn(problem, X, **kw)

    def repairfn(self, problem, X, **kw):
        pass


# repair implementations
# ----------
class WaypointsInfillRepair(RepairBase):
    """Repair routes by infilling them with equi-distant points when
    adjacent points are farther than the specified distance resolution (gcr_dist)
    """

    def repairfn(self, problem, X, **kw):
        gcr_dist = 1e5
        patchfn = patcher.GreatCircleRoutePatcher(dist=gcr_dist)

        for i, (rt,) in enumerate(X):
            route = []

            for j in range(rt.shape[0] - 1):
                p1, p2 = rt[[j, j+1]]
                d = utils.gcr_distance(p1, p2)

                # patch with new waypoints if the distance between any 2
                # adjacent points is greater than 2x the gcr_dist (resolution)

                if not d >= gcr_dist * 2:
                    route.append([p1])
                else:
                    route.append(
                        patchfn.patch(p1, p2, self.config.DEPARTURE_TIME)[:-1])

            route.append(rt[[-1]])
            X[i, 0] = np.concatenate(route, axis=0)
        return X


class ConstraintViolationRepair(RepairBase):
    """Repair routes by finding a feasible route between constraint violations
    """

    def __init__(self, config: Config, constraints_list, **kw):
        super().__init__(config=config)

        self.constraints_list = constraints_list

    def repairfn(self, problem, X, **kw):
        # gcr_dist = 1e5
        # patchfn = patcher.GreatCircleRoutePatcher(dist=gcr_dist)

        patchfn = patcher.IsofuelPatcher.for_single_route(config=self.config)

        for i, (rt,) in enumerate(X):
            constrained = utils.get_constraints_array(rt, self.constraints_list)
            nr = [rt[[0]]]

            p = 0

            for j in range(1, constrained.shape[0]):
                v = constrained[j]

                if v != 0:
                    continue

                if j - p > 1:
                    _r = patchfn.patch(rt[p], rt[j], self.config.DEPARTURE_TIME)
                    nr.append(_r[1:])
                else:
                    nr.append(rt[[j]])
                p = j

            X[i, 0] = np.concatenate(nr, axis=0)
        return X


# orchestration
# ----------
class ChainedRepairsOrchestrator(Repair):
    def __init__(self, order):
        super().__init__()

        self.order = order

    def _do(self, problem, X, **kw):
        for rep in self.order:
            X = rep._do(problem, X, **kw)

        return X


# factory
# ----------
class RepairFactory:
    @staticmethod
    def get_repair(config: Config, constraints_list: ConstraintsList):
        return ChainedRepairsOrchestrator(
            order=[
                WaypointsInfillRepair(config),
                ConstraintViolationRepair(config, constraints_list)
            ], )
