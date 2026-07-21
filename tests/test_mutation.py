import os
from pathlib import Path

import numpy as np
import pytest

import tests.basic_test_func as basic_test_func
from WeatherRoutingTool.algorithms.genetic.mutation import RandomPlateauMutation
from WeatherRoutingTool.config import Config


@pytest.mark.parametrize("route_length", [40, 41])  # one even and one uneven route length
def test_variable_plateau_size(route_length):
    """
    Test genetic.mutation.RandomPlateauMutation.variable_plateau_size():

    - plateau_size and 2 * plateau_slope add up to a value smaller than route_length / 2
    - plateau_size is always uneven

    The function draws the plateau dimensions randomly, so it is called repeatedly (with a fixed seed) to make sure
    the properties hold for many random draws.
    """
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.isofuel_single_route.json')
    config = Config.assign_config(Path(configpath))
    constraint_list = basic_test_func.generate_dummy_constraint_list()

    mt = RandomPlateauMutation(config=config, constraints_list=constraint_list)

    np.random.seed(1)
    for _ in range(100):
        mt.variable_plateau_size(route_length)

        assert mt.plateau_size + 2 * mt.plateau_slope - 2 < np.floor(0.9 * route_length)
        assert mt.plateau_size % 2 == 1
        assert isinstance(mt.plateau_size, int)
        assert isinstance(mt.plateau_slope, int)
