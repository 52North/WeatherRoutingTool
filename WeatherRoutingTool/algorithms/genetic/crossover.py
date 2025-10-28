from pymoo.core.crossover import Crossover

import numpy as np

from datetime import datetime
import logging
import random

from WeatherRoutingTool.constraints.constraints import ConstraintsList
from WeatherRoutingTool.algorithms.genetic import utils
from WeatherRoutingTool.config import Config
from WeatherRoutingTool.algorithms.genetic import patcher

from WeatherRoutingTool.algorithms.genetic.patcher import PatchFactory

logger = logging.getLogger("WRT.genetic.crossover")


# base classes
# ----------
class CrossoverBase(Crossover):
    """Base Crossover Class

    Kept for consistency and to provide super-class level management of
    Crossover implementations.
    """

    def __init__(self, prob=.5):
        super().__init__(n_parents=2, n_offsprings=2, prob=prob)


class OffspringRejectionCrossover(CrossoverBase):
    """Offspring-Rejection Crossover Base Class

    - Generate offsprings using sub-class' implementation of the `crossover` function
    - Validate if offsprings violate discrete constraints
        - if True, get rid of both offsprings, and return parents
        - if False, return offsprings

    :param departure_time: Time of ship departure (from config)
    :type departure_time: datetime
    :param constraints_list: List of constraints
    :type constraints_list: ConstraintsList
    :param Nof_crossover_tries: counter for number of crossover tries
    :type Nof_crossover_tries: int
    :param Nof_crossover_success: counter for number of successful crossovers
    :type Nof_crossover_success: int
    :param crossover_type: crossover type
    :type crossover_type: str
    """

    departure_time: datetime
    constraints_list: ConstraintsList
    config: Config

    Nof_crossover_tries: int
    Nof_crossover_success: int
    crossover_type: str

    constraints_rejection: bool

    def __init__(
            self,
            config: Config,
            departure_time: datetime,
            constraints_list: ConstraintsList,
            prob=.5,
            crossover_type="unnamed"
    ):
        super().__init__(prob=prob)

        self.departure_time = departure_time
        self.constraints_list = constraints_list
        self.Nof_crossover_tries = 0
        self.Nof_crossover_success = 0
        self.crossover_type = crossover_type
        self.constraints_rejection = True
        self.config = config

        if not (config.GENETIC_REPAIR_TYPE == ["no_repair"]):
            self.constraints_rejection = False

    def print_crossover_statistics(self):
        logger.info(f'{self.crossover_type} statistics:')
        logger.info('Nof_crossover_tries: ' + str(self.Nof_crossover_tries))
        logger.info('Nof_crossover_success: ' + str(self.Nof_crossover_success))

    def _do(self, problem, X, **kw):
        # n_parents assumed to be 2
        # n_var assumed to be 1 -> expands into (N, 2)
        n_parents, n_matings, n_var = X.shape

        Y = np.full_like(X, None, dtype=object)

        for k in range(n_matings):
            self.Nof_crossover_tries += 1

            p1 = X[0, k, 0]
            p2 = X[1, k, 0]

            o1, o2 = self.crossover(p1.copy(), p2.copy())

            if ((utils.get_constraints(o1, self.constraints_list) or
                 utils.get_constraints(o2, self.constraints_list)) and
                    self.constraints_rejection):
                Y[0, k, 0] = p1
                Y[1, k, 0] = p2
            else:
                Y[0, k, 0] = o1
                Y[1, k, 0] = o2

                self.Nof_crossover_success += 1
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
    """Single-point Crossover

    :param config: Configuration for the run
    :type config: Config
    """

    def __init__(self, patch_type: str = "gcr", **kw):
        super().__init__(**kw)
        self.patch_type = patch_type

    def crossover(self, p1, p2):
        # setup patching
        patchfn = PatchFactory.get_patcher(patch_type=self.patch_type, config=self.config, application="SP crossover")

        p1x = np.random.randint(1, p1.shape[0] - 1)
        p2x = np.random.randint(1, p2.shape[0] - 1)

        r1 = np.concatenate([
            p1[:p1x],
            patchfn.patch(tuple(p1[p1x - 1]), tuple(p2[p2x]), self.departure_time),
            p2[p2x:], ])

        r2 = np.concatenate([
            p2[:p2x],
            patchfn.patch(tuple(p2[p2x - 1]), tuple(p1[p1x]), self.departure_time),
            p1[p1x:], ])

        return r1, r2


class TwoPointCrossover(OffspringRejectionCrossover):
    """Two-point Crossover

    :param config: Configuration for the run
    :type config: Config
    """

    def __init__(self, patch_type: str = "gcr", **kw):
        super().__init__(**kw)

        self.patch_type = patch_type

    def crossover(self, p1, p2):
        patchfn = PatchFactory.get_patcher(patch_type=self.patch_type, config=self.config, application="TP crossover")

        p1x1 = np.random.randint(1, p1.shape[0] - 4)
        p1x2 = p1x1 + np.random.randint(3, p1.shape[0] - p1x1 - 1)

        p2x1 = np.random.randint(1, p2.shape[0] - 4)
        p2x2 = p2x1 + np.random.randint(3, p2.shape[0] - p2x1 - 1)

        r1 = np.concatenate([
            p1[:p1x1],
            patchfn.patch(tuple(p1[p1x1 - 1]), tuple(p2[p2x1]), self.departure_time),
            p2[p2x1:p2x2],
            patchfn.patch(tuple(p2[p2x2]), tuple(p1[p1x2]), self.departure_time),
            p1[p1x2:], ])

        r2 = np.concatenate([
            p2[:p2x1],
            patchfn.patch(tuple(p2[p2x1 - 1]), tuple(p1[p1x1]), self.departure_time),
            p1[p1x1:p1x2],
            patchfn.patch(tuple(p1[p1x2 - 1]), tuple(p2[p2x2]), self.departure_time),
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


#
# ----------
class RandomizedCrossoversOrchestrator(CrossoverBase):
    """Randomly selects one of the provided crossovers during every call to _do

    :param opts: List of Crossover operators
    :type opts: list[Crossover]
    """

    def __init__(self, opts, **kw):
        super().__init__(**kw)

        self.opts = opts

    def _do(self, problem, X, **kw):
        opt = self.opts[np.random.randint(0, len(self.opts))]
        return opt._do(problem, X, **kw)

    def print_crossover_statistics(self):
        for opt in self.opts:
            opt.print_crossover_statistics()


# factory
# ----------
class CrossoverFactory:
    @staticmethod
    def get_crossover(config: Config, constraints_list: ConstraintsList):
        # inputs
        departure_time = config.DEPARTURE_TIME

        return RandomizedCrossoversOrchestrator(
            opts=[
                TwoPointCrossover(
                    config=config,
                    patch_type=config.GENETIC_CROSSOVER_PATCHER + "_singleton",
                    departure_time=departure_time,
                    constraints_list=constraints_list,
                    prob=.5,
                    crossover_type="TP crossover", ),
                SinglePointCrossover(
                    config=config,
                    patch_type=config.GENETIC_CROSSOVER_PATCHER + "_singleton",
                    departure_time=departure_time,
                    constraints_list=constraints_list,
                    prob=.5,
                    crossover_type="SP crossover", ),
            ], )
