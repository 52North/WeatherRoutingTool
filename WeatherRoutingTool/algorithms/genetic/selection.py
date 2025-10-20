import logging
import math

import numpy as np
from geographiclib.geodesic import Geodesic
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.core.selection import Selection

from WeatherRoutingTool.config import Config

logger = logging.getLogger("WRT.genetic.mutation")


# base class
# ----------
class SelectionBase(Selection):
    """Base Selection Class

    Kept for consistency and to provide super-class level management of
    Mutation implementations.
    """

    def _do(self, _, pop, n_select, n_parents=1, **kwargs):
        pass


class RandomTournamentSelection(SelectionBase):
    rnd_perc: float

    def __init__(self, rnd_perc=0.3):
        self.rnd_perc = rnd_perc

    def _do(self, _, pop, n_select, n_parents=1, **kwargs):
        n_rndm=round(n_select*0.3)
        n_tournamend=n_select-n_rndm

        print('N_select: ', n_select)
        print('N_rndm: ', n_rndm)
        print('N_tournament: ', n_tournamend)

        rs = RandomSelection()
        P_rndm = rs._do(_, pop, n_rndm, n_parents)
        ts = TournamentSelection
        P_tourn = ts._do(_, pop, n_tournamend, n_parents)

        print('P_rndm: ', P_rndm)
        print('P_tourn: ', P_tourn)


