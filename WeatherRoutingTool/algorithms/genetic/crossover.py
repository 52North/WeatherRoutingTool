import logging
import random
from copy import deepcopy
from datetime import datetime
from math import ceil

import numpy as np
from geographiclib.geodesic import Geodesic
from pymoo.core.crossover import Crossover

from WeatherRoutingTool.constraints.constraints import ConstraintsList
from WeatherRoutingTool.algorithms.genetic import utils
from WeatherRoutingTool.config import Config
from WeatherRoutingTool.algorithms.genetic.patcher import PatchFactory

logger = logging.getLogger("WRT.genetic.crossover")

geod = Geodesic.WGS84


# base classes
# ----------
class CrossoverBase(Crossover):
    """Base Crossover Class.

    Kept for consistency and to provide super-class level management of
    Crossover implementations.
    """

    def __init__(self, prob=.5):
        super().__init__(n_parents=2, n_offsprings=2, prob=prob)


class OffspringRejectionCrossover(CrossoverBase):
    """Offspring-Rejection Crossover Base Class.

    - generate offsprings using sub-class' implementation of ``crossover`` function,
    - rejects offspring that violates constraints based on the config variable ``GENETIC_REPAIR_TYPE``

        - if ``GENETIC_REPAIR_TYPE="no_repair"``, ``constraints_rejection`` is set to ``True`` and offspring that
          violates constraints is rejected such that the parents are returned,
        - if ``GENETIC_REPAIR_TYPE`` is set to any valid repair strategy, ``constraints_rejection`` is set to ``False``
          and all crossover candidates are accepted,
    - counts the number of tried and successful crossovers.

    :param departure_time: Time of ship departure (from config).
    :type departure_time: datetime
    :param constraints_list: List of constraints.
    :type constraints_list: ConstraintsList
    :param nof_crossover_tries: Counter for number of crossover tries.
    :type nof_crossover_tries: int
    :param nof_crossover_success: Counter for number of successful crossovers.
    :type nof_crossover_success: int
    :param crossover_type: Crossover type.
    :type crossover_type: str
    :param constraints_rejection: If ``True``, crossover candidates that violate constraints are rejected. If ``False``,
        all crossover candidates are accepted. The variable is set based on config variable ``GENETIC_REPAIR_TYPE``.
        Defaults to ``True``.
    :type constraints_rejection: bool
    """

    departure_time: datetime
    constraints_list: ConstraintsList
    config: Config

    nof_crossover_tries: int
    nof_crossover_success: int
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
        self.nof_crossover_tries = 0
        self.nof_crossover_success = 0
        self.crossover_type = crossover_type
        self.constraints_rejection = True
        self.config = config

        if not (config.GENETIC_REPAIR_TYPE == ["no_repair"]):
            self.constraints_rejection = False

    def print_crossover_statistics(self):
        logger.info(f'{self.crossover_type} statistics:')
        logger.info('nof_crossover_tries: ' + str(self.nof_crossover_tries))
        logger.info('nof_crossover_success: ' + str(self.nof_crossover_success))

    def _do(self, problem, X, **kw):
        # n_parents assumed to be 2
        # n_var assumed to be 1 -> expands into (N, 2)
        n_parents, n_matings, n_var = X.shape

        Y = np.full_like(X, None, dtype=object)

        for k in range(n_matings):
            self.nof_crossover_tries += 1

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

                self.nof_crossover_success += 1
        return Y

    def crossover(
            self,
            p1: np.ndarray,
            p2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Subclass' implementation of the crossover function."""
        return p1, p2

    def route_constraint_violations(self, route: np.ndarray) -> np.ndarray:
        """Check if route breaks any discrete constraints.

        :param route: List of waypoints.
        :type route: np.ndarray
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
    """Single-point Crossover.

    :param patch_type: Type of patcher. Defaults to patching with ``GreatCircleRoutePatcherSingleton``.
    :type patch_type: str
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
    """Two-point Crossover.

    :param patch_type: Type of patcher. Defaults to patching with ``GreatCircleRoutePatcherSingleton``.
    :type patch_type: str
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


#
# ----------
class RandomizedCrossoversOrchestrator(CrossoverBase):
    """Randomly selects one of the provided crossovers during every call of ``_do``.

    :param opts: List of Crossover operators.
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


class SpeedCrossover(OffspringRejectionCrossover):
    """
    Crossover class for ship speed
    """

    def __init__(self, **kw):
        # for now, we don't want to allow repairing routes for speed crossover
        config = deepcopy(kw['config'])
        config.GENETIC_REPAIR_TYPE = ["no_repair"]
        kw['config'] = config
        super().__init__(**kw)
        self.threshold = 50000  # in m
        self.percentage = 0.5

    def crossover(
            self,
            p1: np.ndarray,
            p2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # Find points between parents with a distance below the specified threshold.
        # There should always be one candidate (source). The destination has to be ignored.
        crossover_candidates = []
        for m in range(0, len(p1)-1):
            coord1 = p1[m, 0:2]
            for n in range(0, len(p2)-1):
                coord2 = p2[n, 0:2]
                d = geod.Inverse(coord1[0], coord1[1], coord2[0], coord2[1])["s12"]
                if d < self.threshold:
                    crossover_candidates.append((m, n))
        # Swap speed values for a subset of candidate points
        indices = random.sample(range(0, len(crossover_candidates)), ceil(self.percentage*len(crossover_candidates)))
        for idx in indices:
            c = crossover_candidates[idx]
            speed1 = p1[c[0], -1]
            p1[c[0], -1] = p2[c[1], -1]
            p2[c[1], -1] = speed1
        return p1, p2


class NoCrossover(CrossoverBase):
    """
    Crossover class for ship speed
    """

    def __init__(self, **kw):
        super().__init__()

    def _do(self, problem, X, **kw):
        return X

    def print_crossover_statistics(self):
        pass


# factory
# ----------
class CrossoverFactory:
    @staticmethod
    def get_crossover(config: Config, constraints_list: ConstraintsList):
        departure_time = config.DEPARTURE_TIME

        # FIXME: add exception for bad combinations (better do this on the Config)

        if config.GENETIC_CROSSOVER_TYPE == "no_crossover":
            logger.debug('Setting crossover type of genetic algorithm to "no_crossover".')
            return NoCrossover()

        if config.GENETIC_CROSSOVER_TYPE == "speed":
            logger.debug('Setting crossover type of genetic algorithm to "speed".')
            return SpeedCrossover(
                config=config,
                departure_time=departure_time,
                constraints_list=constraints_list,
                prob=.5,
                crossover_type="Speed crossover")

        if config.GENETIC_CROSSOVER_TYPE == "random":
            logger.debug('Setting crossover type of genetic algorithm to "random".')
            return RandomizedCrossoversOrchestrator(
                opts=[
                    TwoPointCrossover(
                        config=config,
                        patch_type=config.GENETIC_CROSSOVER_PATCHER + "_singleton",
                        departure_time=departure_time,
                        constraints_list=constraints_list,
                        prob=.5,
                        crossover_type="TP crossover"),
                    SinglePointCrossover(
                        config=config,
                        patch_type=config.GENETIC_CROSSOVER_PATCHER + "_singleton",
                        departure_time=departure_time,
                        constraints_list=constraints_list,
                        prob=.5,
                        crossover_type="SP crossover")
                ])
