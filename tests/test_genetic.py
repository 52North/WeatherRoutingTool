import copy
import os
from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

import tests.basic_test_func as basic_test_func
import WeatherRoutingTool.utils.graphics as graphics
from WeatherRoutingTool.algorithms.genetic.patcher import PatcherBase, GreatCircleRoutePatcher, IsofuelPatcher, \
    GreatCircleRoutePatcherSingleton, IsofuelPatcherSingleton
from WeatherRoutingTool.algorithms.genetic.mutation import RandomPlateauMutation, RouteBlendMutation
from WeatherRoutingTool.config import Config
from WeatherRoutingTool.ship.ship_config import ShipConfig
from WeatherRoutingTool.utils.maps import Map


def test_isofuelpatcher_singleton():
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.isofuel_single_route.json')
    config = Config.assign_config(Path(configpath))
    src = [38.851, 4.066]
    dst = [37.901, 8.348]

    departure_time = datetime(2025, 4, 1, 12, 11)
    pt_one = IsofuelPatcherSingleton(config)
    pt_two = IsofuelPatcherSingleton(config)

    pt_one.patch(src, dst, departure_time)

    assert id(pt_two) == id(pt_one)


def test_isofuelpatcher_no_singleton():
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.isofuel_single_route.json')
    config = Config.assign_config(Path(configpath))
    src = [38.851, 4.066]
    dst = [37.901, 8.348]

    departure_time = datetime(2025, 4, 1, 12, 11)
    pt_one = IsofuelPatcher(config)
    pt_two = IsofuelPatcher(config)

    pt_one.patch(src, dst, departure_time)

    assert id(pt_two) != id(pt_one)


def get_dummy_route_input(length='long'):
    route1 = np.array([
        [35.199, 15.490],
        [34.804, 16.759],
        [34.447, 18.381],
        [34.142, 18.763],
        [33.942, 21.080],
        [33.542, 23.024],
        [33.408, 24.389],
        [33.166, 26.300],
        [32.937, 27.859],
        [32.737, 28.859],
    ])
    route2 = np.array([
        [35.199, 16.490],
        [34.804, 17.759],
        [34.447, 19.381],
        [34.142, 19.763],
        [33.942, 22.080],
        [33.542, 23.024],
        [33.408, 24.389],
        [33.166, 25.300],
        [32.937, 26.859],
        [32.737, 27.859],
    ])
    if length == "short":
        route1 = np.delete(route1, -1, 0)
        route2 = np.delete(route2, -1, 0)
        route1 = np.delete(route1, -1, 0)
        route2 = np.delete(route2, -1, 0)

    X = np.array([[route1], [route2]])

    return X


'''
   sanity test for output for genetic.mutation.RandomPlateauMutation.mutate():
   - does the shape of the output route matrix resemble the shape of the input route matrix
   - do the starting and end points of all routes match with the input routes
'''


def test_random_plateau_mutation():
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.isofuel_single_route.json')
    config = Config.assign_config(Path(configpath))
    default_map = Map(32., 15, 36, 29)
    input_crs = ccrs.PlateCarree()
    constraint_list = basic_test_func.generate_dummy_constraint_list()
    np.random.seed(1)

    mt = RandomPlateauMutation(config=config, constraints_list=constraint_list)
    mt.dist = 1e5
    X = get_dummy_route_input()
    old_route = copy.deepcopy(X)
    new_route = mt._do(None, X, )

    # plot figure with original and mutated routes
    fig, ax = graphics.generate_basemap(
        map=default_map.get_var_tuple(),
        depth=None,
        start=(35.199, 15.490),
        finish=(32.937, 27.859),
        title='',
        show_depth=False,
        show_gcr=False
    )
    ax.plot(old_route[0, 0][:, 1], old_route[0, 0][:, 0], color="firebrick", transform=input_crs)
    ax.plot(new_route[0, 0][:, 1], new_route[0, 0][:, 0], color="blue", transform=input_crs)
    ax.plot(old_route[1, 0][:, 1], old_route[1, 0][:, 0], color="firebrick", transform=input_crs)
    ax.plot(new_route[1, 0][:, 1], new_route[1, 0][:, 0], color="blue", transform=input_crs)

    assert old_route.shape == new_route.shape
    for i_route in range(old_route.shape[0]):
        assert np.array_equal(old_route[i_route, 0][-1, :], new_route[i_route, 0][-1, :])
        assert np.array_equal(old_route[i_route, 0][0, :], new_route[i_route, 0][0, :])


'''
    test whether routes are returned as they are by genetic.mutation.RandomPlateauMutation.mutate() if they are too
    short for random plateau mutation
'''


def test_random_plateau_mutation_refusal():
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.isofuel_single_route.json')
    config = Config.assign_config(Path(configpath))
    constraint_list = basic_test_func.generate_dummy_constraint_list()

    np.random.seed(1)

    mt = RandomPlateauMutation(config=config, constraints_list=constraint_list)
    X = get_dummy_route_input(length="short")
    old_route = copy.deepcopy(X)
    new_route = mt._do(None, X, )

    assert np.array_equal(old_route, new_route)


'''
   sanity test for output for genetic.mutation.RouteBlendMutation.mutate():
   - does the shape of the output route matrix resemble the shape of the input route matrix
   - do the starting and end points of all routes match with the input routes
'''


def test_bezier_curve_mutation():
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.isofuel_single_route.json')
    config = Config.assign_config(Path(configpath))
    default_map = Map(32., 15, 36, 29)
    input_crs = ccrs.PlateCarree()
    constraint_list = basic_test_func.generate_dummy_constraint_list()
    np.random.seed(2)

    mt = RouteBlendMutation(config=config, constraints_list=constraint_list)
    X = get_dummy_route_input()
    old_route = copy.deepcopy(X)
    new_route = mt._do(None, X, )

    # plot figure with original and mutated routes
    fig, ax = graphics.generate_basemap(
        map=default_map.get_var_tuple(),
        depth=None,
        start=(35.199, 15.490),
        finish=(32.737, 28.859),
        title='',
        show_depth=False,
        show_gcr=False
    )

    ax.plot(old_route[0, 0][:, 1], old_route[0, 0][:, 0], color="firebrick", transform=input_crs)
    ax.plot(new_route[0, 0][:, 1], new_route[0, 0][:, 0], color="blue", transform=input_crs)
    ax.plot(old_route[1, 0][:, 1], old_route[1, 0][:, 0], color="firebrick", transform=input_crs)
    ax.plot(new_route[1, 0][:, 1], new_route[1, 0][:, 0], color="blue", transform=input_crs)

    assert old_route.shape == new_route.shape
    for i_route in range(old_route.shape[0]):
        assert np.array_equal(old_route[i_route, 0][-1, :], new_route[i_route, 0][-1, :])
        assert np.array_equal(old_route[i_route, 0][0, :], new_route[i_route, 0][0, :])


'''
    test whether routes are returned as they are by genetic.mutation.RouteBlendMutation.mutate() if they are too
    short for route-blend mutation
'''


def test_bezier_mutation_refusal():
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.isofuel_single_route.json')
    config = Config.assign_config(Path(configpath))
    constraint_list = basic_test_func.generate_dummy_constraint_list()

    np.random.seed(1)

    mt = RouteBlendMutation(config=config, constraints_list=constraint_list)
    mt.min_length = 9
    X = get_dummy_route_input(length="short")
    old_route = copy.deepcopy(X)
    new_route = mt._do(None, X, )

    assert np.array_equal(old_route, new_route)


'''
    test whether configuration parameters relevant for the constraint module are not overwritten by config files for
    IsofuelPatcher
'''


def test_configuration_isofuel_patcher():
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.isofuel_single_route.json')
    config = Config.assign_config(Path(configpath))
    config_ship = ShipConfig.assign_config(Path(configpath))

    pt = IsofuelPatcher(base_config=config)

    # check correct configuration of ship parameters
    assert config_ship.BOAT_DRAUGHT_AFT * u.meter == pt.boat.draught_aft
    assert config_ship.BOAT_DRAUGHT_FORE * u.meter == pt.boat.draught_fore
    assert config_ship.BOAT_UNDER_KEEL_CLEARANCE * u.meter == pt.boat.under_keel_clearance
