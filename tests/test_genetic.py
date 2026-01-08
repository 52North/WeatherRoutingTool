import copy
import os
from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

import tests.basic_test_func as basic_test_func
import WeatherRoutingTool.utils.graphics as graphics
from WeatherRoutingTool.algorithms.genetic.crossover import SinglePointCrossover
from WeatherRoutingTool.algorithms.genetic.patcher import PatcherBase, GreatCircleRoutePatcher, IsofuelPatcher, \
    GreatCircleRoutePatcherSingleton, IsofuelPatcherSingleton, PatchFactory
from WeatherRoutingTool.algorithms.genetic.mutation import RandomPlateauMutation, RouteBlendMutation
from WeatherRoutingTool.config import Config
from WeatherRoutingTool.algorithms.genetic.repair import ConstraintViolationRepair
from WeatherRoutingTool.ship.ship_config import ShipConfig
from WeatherRoutingTool.utils.maps import Map

# FIXME: the following test functions fail if LaTeX is not installed:
#   - tests/test_genetic.py::test_random_plateau_mutation
#   - tests/test_genetic.py::test_bezier_curve_mutation
#   - tests/test_genetic.py::test_constraint_violation_repair
#  In the GH Actions workflow, we install the packages texlive, texlive-latex-extra and cm-super to make sure the
#  tests are passing. However, this leads to additional traffic when running the workflow. It would be better to
#  exclude plotting in the tests or adapt it so that LaTeX doesn't need to be installed.


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
        [35.199, 15.490, 10],
        [34.804, 16.759, 10],
        [34.447, 18.381, 10],
        [34.142, 18.763, 10],
        [33.942, 21.080, 10],
        [33.542, 23.024, 10],
        [33.408, 24.389, 10],
        [33.166, 26.300, 10],
        [32.937, 27.859, 10],
        [32.737, 28.859, 10],
    ])
    route2 = np.array([
        [35.199, 16.490, 20],
        [34.804, 17.759, 20],
        [34.447, 19.381, 20],
        [34.142, 19.763, 20],
        [33.942, 22.080, 20],
        [33.542, 23.024, 20],
        [33.408, 24.389, 20],
        [33.166, 25.300, 20],
        [32.937, 26.859, 20],
        [32.737, 27.859, 20],
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

def get_route_lc(X):
    lats = X[:, 0]
    lons = X[:, 1]
    speed = X[:, 2]

    points = np.array([lons, lats]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = Normalize(vmin=10, vmax=20)
    lc = LineCollection(segments, cmap='viridis', norm=norm, transform=ccrs.Geodetic())
    lc.set_array(speed)
    lc.set_linewidth(3)
    return lc


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
    old_route_one_lc = get_route_lc(old_route[0,0])
    old_route_two_lc = get_route_lc(old_route[1,0])
    new_route_one_lc = get_route_lc(new_route[0,0])
    new_route_two_lc = get_route_lc(new_route[1,0])
    ax.add_collection(old_route_one_lc)
    ax.add_collection(old_route_two_lc)
    ax.add_collection(new_route_one_lc)
    ax.add_collection(new_route_two_lc)

    cbar = fig.colorbar(old_route_one_lc, ax=ax, orientation='vertical', pad=0.15, shrink=0.7)
    cbar.set_label('Geschwindigkeit ($m/s$)')

    plt.tight_layout()
    plt.show()


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

    old_route_one_lc = get_route_lc(old_route[0,0])
    old_route_two_lc = get_route_lc(old_route[1,0])
    new_route_one_lc = get_route_lc(new_route[0,0])
    new_route_two_lc = get_route_lc(new_route[1,0])
    ax.add_collection(old_route_one_lc)
    ax.add_collection(old_route_two_lc)
    ax.add_collection(new_route_one_lc)
    ax.add_collection(new_route_two_lc)

    cbar = fig.colorbar(old_route_one_lc, ax=ax, orientation='vertical', pad=0.15, shrink=0.7)
    cbar.set_label('Geschwindigkeit ($m/s$)')

    plt.tight_layout()
    plt.show()

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


def test_constraint_violation_repair():
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.isofuel_single_route.json')
    config = Config.assign_config(Path(configpath))
    default_map = Map(32., 15, 36, 29)
    input_crs = ccrs.PlateCarree()
    constraint_list = basic_test_func.generate_dummy_constraint_list()
    np.random.seed(2)

    patchfn = PatchFactory.get_patcher(
        patch_type="isofuel_singleton",
        config=config,
        application="ConstraintViolationRepair"
    )
    repairfn = ConstraintViolationRepair(config, constraint_list)
    X = get_dummy_route_input()
    old_route = copy.deepcopy(X)
    is_constrained = [False, True, True, True, False, True, True, False, False]
    new_route = repairfn.repair_single_route(X[0, 0], patchfn, is_constrained)

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
    old_route_lc = get_route_lc(old_route[0, 0])
    new_route_lc = get_route_lc(new_route)
    ax.add_collection(old_route_lc)
    ax.add_collection(new_route_lc)

    cbar = fig.colorbar(old_route_lc, ax=ax, orientation='vertical', pad=0.15, shrink=0.7)
    cbar.set_label('Geschwindigkeit ($m/s$)')

    plt.tight_layout()
    plt.show()

    assert np.array_equal(new_route[0], old_route[0, 0][0])
    assert np.array_equal(new_route[-2], old_route[0, 0][-2])
    assert np.array_equal(new_route[-1], old_route[0, 0][-1])


def test_single_point_crossover():
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.isofuel_single_route.json')
    config = Config.assign_config(Path(configpath))
    default_map = Map(32., 15, 36, 29)
    input_crs = ccrs.PlateCarree()
    constraint_list = basic_test_func.generate_dummy_constraint_list()
    departure_time =  datetime(2025, 4, 1, 11, 11)

    np.random.seed(2)

    X = get_dummy_route_input()
    old_route = copy.deepcopy(X)

    sp = SinglePointCrossover(
        config=config,
        constraints_list=constraint_list,
        departure_time=departure_time
    )
    # r1, r2 = sp.crossover(X[0,0], X[1,0])
    X = sp._do(problem=None, X=X)

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

    # ax.plot(r1[:, 1], r1[:, 0], color="blue", transform=input_crs, marker='o')
    # ax.plot(r2[:, 1], r2[:, 0], color="blue", transform=input_crs, marker='o')
    ax.plot(X[0, 0][:, 1], old_route[0, 0][:, 0], color="green", transform=input_crs, marker='o')
    ax.plot(old_route[0, 0][:, 1], old_route[0, 0][:, 0], color="green", transform=input_crs, marker='o')
    ax.plot(old_route[1, 0][:, 1], old_route[0, 0][:, 0], color="orange", transform=input_crs, marker='o')

    plt.show()

    assert 1==2

