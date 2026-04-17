import copy
import os
from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as pyplot
import pytest
from astropy import units as u

import tests.basic_test_func as basic_test_func
import WeatherRoutingTool.algorithms.genetic.utils as utils
import WeatherRoutingTool.utils.graphics as graphics
from WeatherRoutingTool.algorithms.genetic import Genetic
from WeatherRoutingTool.algorithms.genetic.crossover import SinglePointCrossover, SpeedCrossover, TwoPointCrossoverSpeed
from WeatherRoutingTool.algorithms.genetic.patcher import PatcherBase, GreatCircleRoutePatcher, IsofuelPatcher, \
    GreatCircleRoutePatcherSingleton, IsofuelPatcherSingleton, PatchFactory
from WeatherRoutingTool.algorithms.genetic.population import IsoFuelPopulation, FromGeojsonPopulation
from WeatherRoutingTool.algorithms.genetic.mutation import RandomPlateauMutation, RouteBlendMutation
from WeatherRoutingTool.config import Config
from WeatherRoutingTool.algorithms.genetic.repair import ConstraintViolationRepair
from WeatherRoutingTool.ship.ship_config import ShipConfig
from WeatherRoutingTool.utils.maps import Map


def test_isofuelpatcher_singleton():
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.isofuel_single_route.json')
    config = Config.assign_config(Path(configpath))
    src = [38.851, 4.066, 7]
    dst = [37.901, 8.348, 7]

    departure_time = datetime(2025, 4, 1, 12, 11)
    pt_one = IsofuelPatcherSingleton(config)
    pt_two = IsofuelPatcherSingleton(config)

    pt_one.patch(src, dst, departure_time)

    assert id(pt_two) == id(pt_one)


def test_isofuelpatcher_no_singleton():
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.isofuel_single_route.json')
    config = Config.assign_config(Path(configpath))
    src = [38.851, 4.066, 7]
    dst = [37.901, 8.348, 7]

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


@pytest.mark.manual
def test_random_plateau_mutation(plt):
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.isofuel_single_route.json')
    config = Config.assign_config(Path(configpath))
    default_map = Map(32., 15, 36, 29)
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
    old_route_one_lc = graphics.get_route_lc(old_route[0, 0])
    old_route_two_lc = graphics.get_route_lc(old_route[1, 0])
    new_route_one_lc = graphics.get_route_lc(new_route[0, 0])
    new_route_two_lc = graphics.get_route_lc(new_route[1, 0])
    ax.add_collection(old_route_one_lc)
    ax.add_collection(old_route_two_lc)
    ax.add_collection(new_route_one_lc)
    ax.add_collection(new_route_two_lc)

    cbar = fig.colorbar(old_route_one_lc, ax=ax, orientation='vertical', pad=0.15, shrink=0.7)
    cbar.set_label('Geschwindigkeit ($m/s$)')

    pyplot.tight_layout()
    plt.saveas = "test_random_plateau_mutation.png"

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


@pytest.mark.manual
def test_bezier_curve_mutation(plt):
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.isofuel_single_route.json')
    config = Config.assign_config(Path(configpath))
    default_map = Map(32., 15, 36, 29)
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

    old_route_one_lc = graphics.get_route_lc(old_route[0, 0])
    old_route_two_lc = graphics.get_route_lc(old_route[1, 0])
    new_route_one_lc = graphics.get_route_lc(new_route[0, 0])
    new_route_two_lc = graphics.get_route_lc(new_route[1, 0])
    ax.add_collection(old_route_one_lc)
    ax.add_collection(old_route_two_lc)
    ax.add_collection(new_route_one_lc)
    ax.add_collection(new_route_two_lc)

    cbar = fig.colorbar(old_route_one_lc, ax=ax, orientation='vertical', pad=0.15, shrink=0.7)
    cbar.set_label('Geschwindigkeit ($m/s$)')

    pyplot.tight_layout()
    plt.saveas = "test_bezier_curve_mutation.png"

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


@pytest.mark.manual
def test_constraint_violation_repair(plt):
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.isofuel_single_route.json')
    config = Config.assign_config(Path(configpath))
    default_map = Map(32., 15, 36, 29)
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
    old_route_lc = graphics.get_route_lc(old_route[0, 0])
    new_route_lc = graphics.get_route_lc(new_route)
    ax.add_collection(old_route_lc)
    ax.add_collection(new_route_lc)

    cbar = fig.colorbar(old_route_lc, ax=ax, orientation='vertical', pad=0.15, shrink=0.7)
    cbar.set_label('Geschwindigkeit ($m/s$)')

    pyplot.tight_layout()
    plt.saveas = "test_constraint_violation_repair.png"

    assert np.array_equal(new_route[0], old_route[0, 0][0])
    assert np.array_equal(new_route[-2], old_route[0, 0][-2])
    assert np.array_equal(new_route[-1], old_route[0, 0][-1])


def test_recalculate_speed_for_route():
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.isofuel_single_route.json')
    config = Config.assign_config(Path(configpath))
    config.ARRIVAL_TIME = datetime(2025, 4, 2, 11, 11)
    config.DEPARTURE_TIME = datetime(2025, 4, 1, 11, 11)
    constraint_list = basic_test_func.generate_dummy_constraint_list()

    pop = IsoFuelPopulation(
        config=config,
        default_route=[35.199, 15.490, 32.737, 28.859],
        constraints_list=constraint_list,
        pop_size=1
    )
    rt = get_dummy_route_input()
    rt = rt[0, 0]
    new_route = copy.deepcopy(rt)
    new_route = pop.recalculate_speed_for_route(new_route)

    dist_to_dest = 1262000 * u.meter
    time_difference = config.ARRIVAL_TIME - config.DEPARTURE_TIME
    bs_approx = dist_to_dest / (time_difference.total_seconds() * u.second)

    assert np.all((new_route[:, 2] - bs_approx.value) < 0.3)


@pytest.mark.manual
@pytest.mark.skip(reason="Test needs modified route array.")
def test_single_point_crossover(plt):
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.isofuel_single_route.json')
    config = Config.assign_config(Path(configpath))
    default_map = Map(32., 15, 36, 29)
    input_crs = ccrs.PlateCarree()
    constraint_list = basic_test_func.generate_dummy_constraint_list()
    departure_time = datetime(2025, 4, 1, 11, 11)

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

    ax.plot(X[0, 0][:, 1], old_route[0, 0][:, 0], color="green", transform=input_crs, marker='o')
    ax.plot(old_route[0, 0][:, 1], old_route[0, 0][:, 0], color="green", transform=input_crs, marker='o')
    ax.plot(old_route[1, 0][:, 1], old_route[0, 0][:, 0], color="orange", transform=input_crs, marker='o')

    plt.saveas = "test_single_point_crossoverr.png"


@pytest.mark.manual
def test_speed_crossover(plt):
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.isofuel_single_route.json')
    config = Config.assign_config(Path(configpath))
    default_map = Map(32., 15, 36, 29)
    constraint_list = basic_test_func.generate_dummy_constraint_list()
    departure_time = datetime(2025, 4, 1, 11, 11)

    X = get_dummy_route_input()

    sp = SpeedCrossover(config=config, departure_time=departure_time, constraints_list=constraint_list)
    o1, o2 = sp.crossover(X[0, 0], X[1, 0])

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
    old_X1_lc = graphics.get_route_lc(X[0, 0])
    old_X2_lc = graphics.get_route_lc(X[1, 0])

    new_X1_lc = graphics.get_route_lc(o1)
    new_X2_lc = graphics.get_route_lc(o2)

    ax.add_collection(old_X1_lc)
    ax.add_collection(old_X2_lc)
    ax.add_collection(new_X1_lc)
    ax.add_collection(new_X2_lc)

    cbar = fig.colorbar(old_X2_lc, ax=ax, orientation='vertical', pad=0.15, shrink=0.7)
    cbar.set_label('Geschwindigkeit ($m/s$)')

    pyplot.tight_layout()
    plt.saveas = "test_speed_crossover.png"


def test_spread_velocity(plt):
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.isofuel_single_route.json')
    routepath = Path(os.path.join(dirname, 'data/'))
    config = Config.assign_config(Path(configpath))
    default_map = [32., 15, 36, 29]
    constraint_list = basic_test_func.generate_dummy_constraint_list()

    min_boat_speed = 3
    max_boat_speed = 15
    boat_speed = 7
    population_size = 20

    pop = FromGeojsonPopulation(
        config=config,
        default_route=default_map,
        constraints_list=constraint_list,
        pop_size=population_size,
        routes_dir=routepath
    )
    quantiles = pop.spread_velocity(min_boat_speed, max_boat_speed, boat_speed, population_size)

    assert quantiles.shape[0] == population_size
    assert np.min(quantiles) >= min_boat_speed
    assert np.max(quantiles) <= max_boat_speed

    x_dummy = np.full(quantiles.shape, 1)
    fig, ax = pyplot.subplots(figsize=graphics.get_standard('fig_size'))
    marker_quant = dict(
        marker="o",
        markersize=5,
        markerfacecolor="gold",
        markeredgecolor="black", )

    marker_bounds = dict(
        marker="x",
        markersize=7,
        markerfacecolor="blue",
        markeredgecolor="blue", )
    ax.plot(quantiles, x_dummy, **marker_quant, color="none")
    ax.plot([min_boat_speed, max_boat_speed, boat_speed], [1, 1, 1], **marker_bounds, color="none")

    plt.saveas = "test_spread_velocidy.png"


@pytest.mark.parametrize("speed_arr,viol_list", [
    (np.array([1, 2, 3, 4, 5, 6, 7, -99]), []),
    (np.array([1, 4, 3, 4, 8, 7, 6, -99]), [0, 1, 3, 4]),
])
def test_check_speed_dif(speed_arr, viol_list):
    """
    Test whether correct lists is returned from utils.check_speed_dif
    """
    viol_list_test = utils.check_speed_dif(speed_arr, 2)
    assert viol_list_test == viol_list


@pytest.mark.parametrize("speed_arr,", [
    (np.array([1., 2., 100000., 4., 5., 6., 1000., -99])),
])
def test_smoothen_speed_rec_error(speed_arr):
    """
    Test whether exception is raised if utils.smoothen_speed_rec function is called too often.
    """
    with pytest.raises(Exception) as excinfo:
        utils.smoothen_speed(speed_arr, 2)

    assert "Too many calls to smoothen" in str(excinfo.value)


@pytest.mark.parametrize("speed_arr,smooth_res", [
    (np.array([1., 2., 5., 4., 5., 6., 10., -99.]), np.array([1., 2.5, 4., 4., 5., 6.75, 8.6666666, -99.])),
    (np.array([10., 2., 5., 4., 5., 6., 10., -99.]),
     np.array([6.472222, 5.2083333, 4., 4., 5., 6.75, 8.6666666, -99.])),

])
def test_smoothen_speed_success(speed_arr, smooth_res):
    """
    Test whether correct smoothened list is returned from utils.smoothen_speed.
    """
    smooth_arr = utils.smoothen_speed(speed_arr, 2)

    assert np.isclose(smooth_arr, smooth_res).all()


def test_twopoint_crossover_speed(plt):
    """
    Test whether TwoPointCrossoverSpeed provides sensible results via monitoring plot.
    """
    dirname = os.path.dirname(__file__)
    configpath = os.path.join(dirname, 'config.isofuel_single_route.json')
    config = Config.assign_config(Path(configpath))
    default_map = Map(32., 15, 36, 29)
    constraint_list = basic_test_func.generate_dummy_constraint_list()
    departure_time = datetime(2025, 4, 1, 11, 11)

    X = get_dummy_route_input()

    sp = TwoPointCrossoverSpeed(config=config, departure_time=departure_time, constraints_list=constraint_list)
    o1, o2 = sp.crossover(X[0, 0], X[1, 0])

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
    old_X1_lc = graphics.get_route_lc(X[0, 0])
    old_X2_lc = graphics.get_route_lc(X[1, 0])

    new_X1_lc = graphics.get_route_lc(o1)
    new_X2_lc = graphics.get_route_lc(o2)

    ax.add_collection(old_X1_lc)
    ax.add_collection(old_X2_lc)
    ax.add_collection(new_X1_lc)
    ax.add_collection(new_X2_lc)

    cbar = fig.colorbar(old_X2_lc, ax=ax, orientation='vertical', pad=0.15, shrink=0.7)
    cbar.set_label('Geschwindigkeit ($m/s$)')

    pyplot.tight_layout()
    plt.saveas = "test_twopoint_crossover_speed.png"
