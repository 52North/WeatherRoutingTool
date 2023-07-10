import datetime
import os

import xarray
import pytest

from algorithms.isobased import IsoBased
from algorithms.isofuel import IsoFuel
from constraints.constraints import *

def generate_dummy_constraint_list():
    pars = ConstraintPars()
    pars.resolution = 1./10

    constraint_list = ConstraintsList(pars)
    return constraint_list

def create_dummy_IsoBased_object():
    start = (30, 45)
    finish = (0, 20)
    date = datetime.date.today()
    prune_sector_half = 90
    nof_prune_segments = 5
    nof_hdgs_segments = 4
    hdgs_increments = 1

    ra = IsoBased(start, finish, date)
    ra.set_pruning_settings(prune_sector_half, nof_prune_segments)
    ra.set_variant_segments(nof_hdgs_segments, hdgs_increments)
    return ra

def create_dummy_IsoFuel_object():
    start = (30, 45)
    finish = (0, 20)
    date = datetime.date.today()
    prune_sector_half = 90
    nof_prune_segments = 5
    nof_hdgs_segments = 4
    hdgs_increments = 1

    ra = IsoFuel(start, finish, date, 999, "")
    ra.set_pruning_settings(prune_sector_half, nof_prune_segments)
    ra.set_variant_segments(nof_hdgs_segments, hdgs_increments)
    return ra
