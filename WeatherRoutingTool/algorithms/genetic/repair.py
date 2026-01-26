from pymoo.core.repair import Repair, NoRepair

import numpy as np
import logging

from WeatherRoutingTool.config import Config
from WeatherRoutingTool.constraints.constraints import ConstraintsList
from WeatherRoutingTool.algorithms.genetic import utils
from WeatherRoutingTool.algorithms.genetic.patcher import PatchFactory

logger = logging.getLogger("WRT.genetic.repair")


# ----------
class RepairBase(Repair):
    """Base Repair class

    Kept for consistency and to provide super-class level management of
    Repair implementations.

    :param config: Configuration for the run
    :type config: Config
    """

    config: Config

    def __init__(self, config: Config):
        super().__init__()

        self.config = config

    def _do(self, problem, X, **kw):
        return self.repairfn(problem, X, **kw)

    def repairfn(self, problem, X, **kw):
        """Implemenation of the repair function"""

        pass


# repair implementations
# ----------
class WaypointsInfillRepair(RepairBase):
    """Repair routes by infilling them with equi-distant points when
    adjacent points are farther than the specified distance resolution (gcr_dist)
    """

    def repairfn(self, problem, X, **kw):
        gcr_dist = 1e5
        patchfn = PatchFactory.get_patcher(patch_type="gcr", application="WaypointsInfillRepair")

        for i, (rt,) in enumerate(X):
            route = []

            for j in range(rt.shape[0] - 1):
                p1, p2 = rt[[j, j + 1]]
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

    :param config: Configuration for the run
    :type config: Config
    """

    def __init__(self, config: Config, constraints_list, **kw):
        super().__init__(config=config)

        self.constraints_list = constraints_list

    def repairfn(self, problem, X, **kw):
        """
        Request constraint validation and repairing of routes.

        :param X: Route matrix in the form of ``np.array([[route_0], [route_1], ...])`` with
            ``route_i=np.array([[lat_0, lon_0], [lat_1,lon_1], ...])``. X.shape = (n_routes, 1, n_waypoints, 2).
            Access i'th route as ``X[i,0]`` and the j'th coordinate pair off the i'th route as ``X[i,0][j, :]``.
        :type X: np.array
        :return: Repaired route matrix. Same structure as for ``X``.
        :rtype: np.array
        """

        patchfn = PatchFactory.get_patcher(
            patch_type="isofuel_singleton",
            config=self.config,
            application="ConstraintViolationRepair"
        )

        for i, (rt,) in enumerate(X):
            constrained = utils.get_constraints_array(rt, self.constraints_list)
            X[i, 0] = self.repair_single_route(rt, patchfn, constrained)

        return X

    def check_validity(self, rt):
        src = tuple(self.config.DEFAULT_ROUTE[:-2])
        dst = tuple(self.config.DEFAULT_ROUTE[-2:])
        assert tuple(rt[0]) == src, "Source waypoint not matching"
        assert tuple(rt[-1]) == dst, "Destination waypoint not matching"

    def repair_single_route(self, rt, patchfn, constrained):
        """
        Repairing route segments which are violating constraints.

        The function loops through ``constrained``. While looping, it stores the index of the last valid
        waypoint in ``prev_seg_end`` and fills the list ``output_route_segs`` with valid route segments:
            - If a segment is constrained, the variable ``on_constraint`` is set to ``True``.
            - If a segment is not constrained and ``on_constraint==False``, the waypoint at the end of the route segment
              is added to ``output_route_segs``.
            - If a segment is not constrained and ``on_constraint==True``, the ship did just pass a constrained area.
              Thus, the patcher is called to connect the last valid waypoint to the current starting point of the route
              segment. The resulting route segment and the last waypoint of the current segment are added to
              ``output_route_segs``.


        :params rt: Route to be repaired.
        :type rt: np.array([[lat_0, lon_0], [lat_1,lon_1], ...]),
        :params patchfn: route Patcher
        :type patchfn: PatcherBase
        :params constrained: Results of checks for constraint violations for each route segment.
        :type constrained: np.array
        :return: Repaired route. Same structure as for ``rt``.
        :rtype: np.array
        """
        # check for correct input shape of rt
        assert len(rt.shape) == 2
        assert rt.shape[1] == 2
        debug = False

        prev_seg_end = 0
        on_constraint = False
        output_route_segs = [np.array([rt[0]])]

        if debug:
            print('rt: ', rt)
            print('output_route_segs: ', output_route_segs)

        for seg_i in range(0, len(constrained)):
            seg_start = seg_i
            seg_end = seg_i + 1

            if debug:
                print('seg_start: ', seg_start)
                print('seg_end: ', seg_end)
                print('on_constrained: ', on_constraint)
                print('constrained: ', constrained[seg_i])
                print('prev_constraint: ', prev_seg_end)
                print('')

            if not constrained[seg_i]:
                if on_constraint:
                    # just passed constrained area -> call patcher
                    _r = patchfn.patch(rt[prev_seg_end], rt[seg_start], self.config.DEPARTURE_TIME)
                    if debug:
                        print(' adding _r: ', _r)

                    output_route_segs.append(_r)
                    output_route_segs.append([rt[seg_end]])
                    on_constraint = False
                    prev_seg_end = seg_end
                else:
                    # on valid route segment
                    if debug:
                        print('rt[seg_end]: ', rt[seg_end])
                    output_route_segs.append([rt[seg_end]])
                    prev_seg_end = seg_end
            else:
                # currently passing constrained area
                on_constraint = True
                if seg_i == len(constrained)-1:
                    output_route_segs.append([rt[seg_end]])

        output_route = np.concatenate(output_route_segs, axis=0)
        if debug:
            print('output_route shape: ', output_route)
            self.check_validity(output_route)
        return output_route


# orchestration
# ----------
class ChainedRepairsOrchestrator(Repair):
    """Executes repairs in the order they are entered in

    :param order: List of Repair implementations to execute in that order
    :type order: list[Repair]"""

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

        if config.GENETIC_REPAIR_TYPE == ["no_repair"]:
            logger.debug('Setting repair type of genetic algorithm to "no_repair".')
            return None

        if "waypoints_infill" in config.GENETIC_REPAIR_TYPE and "constraint_violation" in config.GENETIC_REPAIR_TYPE:
            logger.debug('Setting repair type of genetic algorithm to [waypoints_infill & constraint_violation]')
            return ChainedRepairsOrchestrator(
                order=[
                    WaypointsInfillRepair(config),
                    ConstraintViolationRepair(config, constraints_list)
                ], )

        if "waypoints_infill" in config.GENETIC_REPAIR_TYPE:
            logger.debug('Setting repair type of genetic algorithm to "waypoints_infill".')
            return ChainedRepairsOrchestrator(
                order=[
                    WaypointsInfillRepair(config),
                ], )

        if "constraint_violation" in config.GENETIC_REPAIR_TYPE:
            logger.debug('Setting repair type of genetic algorithm to "constraint_violation".')
            return ChainedRepairsOrchestrator(
                order=[
                    ConstraintViolationRepair(config, constraints_list)
                ], )

        return NotImplementedError(f'The repair type {config.GENETIC_REPAIR_TYPE} is not implemented.')
