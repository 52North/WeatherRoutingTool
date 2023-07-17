import datetime

import pytest
import xarray

import WeatherRoutingTool.utils.graphics as graphics
from WeatherRoutingTool.constraints.constraints import *
from WeatherRoutingTool.ship.shipparams import ShipParams
from WeatherRoutingTool.weather import *

def test_plot_power_vs_dist():
    count = 3
    dummy_scalar = 0
    dummy_list = np.array([])
    fuel_consumed = np.array([1,2,1])
    dist_per_step = np.array([1,4,5])


    sp = ShipParams(fuel_consumed, dummy_list, dummy_list,dummy_list)

    route = RouteParams(
        count = count,  # routing step
        start = dummy_list,  # lat, lon at start
        finish = dummy_list,  # lat, lon at end
        #fuel = dummy_scalar,  # sum of fuel consumption [kWh]
        #rpm = dummy_scalar,  # propeller [revolutions per minute]
        route_type = 'min_time_route',  # route name
        time = dummy_scalar,  # time needed for the route [h]
        lats_per_step=dummy_list,
        lons_per_step=dummy_list,
        azimuths_per_step= dummy_list,
        dists_per_step = dist_per_step,
        #speed_per_step = dummy_list,
        starttime_per_step = dummy_list,
        #fuel_per_step= fuel_consumed,
        #full_dist_traveled = dummy_list,
        ship_params_per_step=sp,
        gcr = dummy_list
    )
    route.plot_power_vs_dist("orange", "Route X")

def test_get_accumulated_dist():
    dist_per_step = np.array([1,4,5,7])
    dist_acc_test = np.array([1,5,10,17])
    dist = graphics.get_accumulated_dist(dist_per_step)

    assert np.array_equal(dist_acc_test, dist)

def test_get_hist_values_from_boundaries():
    bin_boundaries = np.array([1,4,5,7])
    bin_content_unnormalised = np.array([1,3,5])
    bin_content_normalised_test = np.array([1/3, 3, 5/2])
    bin_centres_test = np.array([2.5, 4.5, 6])
    bin_widths_test  = np.array([3, 1, 2])

    hist_values = graphics.get_hist_values_from_boundaries(bin_boundaries, bin_content_unnormalised)

    assert np.array_equal(bin_centres_test, hist_values['bin_centres'])
    assert np.array_equal(bin_widths_test, hist_values['bin_widths'])
    assert np.array_equal(bin_content_normalised_test, hist_values['bin_content'])

def test_get_hist_values_from_widths():
    bin_widths = np.array([3,1,2])
    bin_content_unnormalised = np.array([1,3,5])
    bin_content_normalised_test = np.array([1/3, 3, 5/2])
    bin_centres_test = np.array([1.5, 3.5, 5])

    hist_values = graphics.get_hist_values_from_widths(bin_widths, bin_content_unnormalised)

    assert np.array_equal(bin_centres_test, hist_values['bin_centres'])
    assert np.array_equal(bin_content_normalised_test, hist_values['bin_content'])
