from pymoo.core.crossover import Crossover

import numpy as np

from datetime import datetime
import logging

from WeatherRoutingTool.constraints.constraints import ConstraintsList
from WeatherRoutingTool.algorithms.genetic import utils
from WeatherRoutingTool.config import Config
from WeatherRoutingTool.algorithms.genetic import patcher

logger = logging.getLogger("WRT.genetic.crossover")


# base classes
# ----------
class OffspringRejectionCrossover(Crossover):
    """Offspring-Rejection Crossover Base Class

    Algorithm:
    - Generate offsprings using sub-class' implementation of the `crossover` function
    - Validate if offsprings violate discrete constraints
        - True — get rid of both offsprings, and return parents
        - False — return offsprings
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
                utils.get_constraints(p1, self.constraints_list) or
                utils.get_constraints(p2, self.constraints_list)
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

    def __init__(self, config: Config, **kw):
        super().__init__(**kw)

        # variables
        self.config = config
        self.patch_type = "gcr"

    def crossover(self, p1, p2):
        # setup patching
        match self.patch_type:
            case "isofuel":
                patchfn = patcher.IsofuelPatcher.for_single_route(
                    default_map=self.config.DEFAULT_MAP, )
            case "gcr":
                patchfn = patcher.GreatCircleRoutePatcher(dist=1e5)
            case _:
                raise ValueError("Invalid patcher type")

        p1x = np.random.randint(1, p1.shape[0] - 1)
        p2x = np.random.randint(1, p2.shape[0] - 1)

        r1 = np.concatenate([
            p1[:p1x],
            patchfn.patch(tuple(p1[p1x-1]), tuple(p2[p2x]), self.departure_time),
            p2[p2x:], ])

        r2 = np.concatenate([
            p2[:p2x],
            patchfn.patch(tuple(p2[p2x-1]), tuple(p1[p1x]), self.departure_time),
            p1[p1x:], ])

        return r1, r2


class TwoPointCrossover(OffspringRejectionCrossover):
    """Two-point crossover operator with great-circle patching"""

    def __init__(self, config: Config, **kw):
        super().__init__(**kw)

        # variables
        self.config = config
        self.patch_type = "gcr"

    def crossover(self, p1, p2):
        match self.patch_type:
            case "isofuel":
                patchfn = patcher.IsofuelPatcher.for_single_route(
                    default_map=self.config.DEFAULT_MAP, )
            case "gcr":
                patchfn = patcher.GreatCircleRoutePatcher(dist=1e5)
            case _:
                raise ValueError("Invalid patcher type")

        p1x1 = np.random.randint(1, p1.shape[0] - 4)
        p1x2 = p1x1 + np.random.randint(3, p1.shape[0] - p1x1 - 1)

        p2x1 = np.random.randint(1, p2.shape[0] - 4)
        p2x2 = p2x1 + np.random.randint(3, p2.shape[0] - p2x1 - 1)

        r1 = np.concatenate([
            p1[:p1x1],
            patchfn.patch(tuple(p1[p1x1-1]), tuple(p2[p2x1]), self.departure_time),
            p2[p2x1:p2x2],
            patchfn.patch(tuple(p2[p2x2]), tuple(p1[p1x2]), self.departure_time),
            p1[p1x2:], ])

        r2 = np.concatenate([
            p2[:p2x1],
            patchfn.patch(tuple(p2[p2x1-1]), tuple(p1[p1x1]), self.departure_time),
            p1[p1x1:p1x2],
            patchfn.patch(tuple(p1[p1x2-1]), tuple(p2[p2x2]), self.departure_time),
            p2[p2x2:], ])

        return r1, r2


class PMX(OffspringRejectionCrossover):
    """Partially Mapped Crossover"""

    def crossover(self, p1, p2):
        if p1.shape != p2.shape:
            logging.info("PMX — Not of equal length")
            return p1, p2

        N = min(p1.shape[0], p2.shape[0])

        # Convert to lists of tuples
        parent1 = [tuple(row) for row in p1]
        parent2 = [tuple(row) for row in p2]

        # Choose crossover points
        cx1, cx2 = sorted(np.random.choice(range(N), 2, replace=False))

        # Initialize offspring placeholders
        child1 = [None] * N
        child2 = [None] * N

        # Copy the segment
        for i in range(cx1, cx2):
            child1[i] = parent2[i]
            child2[i] = parent1[i]

        # Build mapping for the swapped segments
        mapping12 = {parent2[i]: parent1[i] for i in range(cx1, cx2)}
        mapping21 = {parent1[i]: parent2[i] for i in range(cx1, cx2)}

        def resolve(gene, segment, mapping):
            # Keep resolving until gene is not in the given segment
            while gene in segment:
                gene = mapping[gene]
            return gene

        # Fill remaining positions
        for i in range(N):
            if not (cx1 <= i < cx2):
                g1 = parent1[i]
                g2 = parent2[i]

                # If g1 is already in the swapped segment of child1, resolve via mapping12
                if g1 in child1[cx1:cx2]:
                    g1 = resolve(g1, child1[cx1:cx2], mapping12)
                child1[i] = g1

                # Likewise for child2
                if g2 in child2[cx1:cx2]:
                    g2 = resolve(g2, child2[cx1:cx2], mapping21)
                child2[i] = g2

        # Convert back to numpy arrays
        c1 = np.array(child1, dtype=p1.dtype)
        c2 = np.array(child2, dtype=p1.dtype)

        return c1, c2


# factory
# ----------
class CrossoverFactory:
    @staticmethod
    def get_crossover(config: Config, constraints_list: ConstraintsList):
        # inputs
        departure_time = config.DEPARTURE_TIME

        return TwoPointCrossover(
            config=config,
            departure_time=departure_time,
            constraints_list=constraints_list,
            prob=.5, )

        return SinglePointCrossover(
            config=config,
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
