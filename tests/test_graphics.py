import datetime

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

import WeatherRoutingTool.utils.graphics as graphics
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.ship.shipparams import ShipParams


def test_plot_power_vs_dist():
    count = 3
    dummy_scalar = 0
    dummy_list = np.full(3, -99)
    dummy_list_coords = np.full(4, -99)
    fuel_consumed = np.array([1, 2, 1]) * u.kg/u.second
    dist_per_step = np.array([1, 4, 5])
    time_per_step = np.array(
        [datetime.datetime(2022, 12, 19),
         datetime.datetime(2022, 12, 19) + datetime.timedelta(days=180),
         datetime.datetime(2022, 12, 19) + datetime.timedelta(days=360),
         datetime.datetime(2022, 12, 19) + datetime.timedelta(days=400)])

    sp = ShipParams(fuel_consumed, dummy_list, dummy_list, dummy_list, dummy_list, dummy_list, dummy_list, dummy_list,
                    dummy_list, dummy_list, dummy_list, dummy_list, dummy_list, dummy_list, dummy_list, dummy_list,
                    dummy_list, dummy_list, dummy_list, dummy_list, dummy_list, dummy_list)

    route = RouteParams(count=count,  # routing step
                        start=dummy_list,  # lat, lon at start
                        finish=dummy_list,  # lat, lon at end
                        route_type='min_time_route',  # route name
                        time=dummy_scalar,  # time needed for the route [h]
                        lats_per_step=dummy_list_coords,
                        lons_per_step=dummy_list_coords,
                        course_per_step=dummy_list * u.degree,
                        dists_per_step=dist_per_step * u.meter,
                        starttime_per_step=time_per_step,
                        ship_params_per_step=sp, gcr=dummy_list)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=96)
    route.plot_power_vs_dist("orange", "Route X", "fuel", ax)


def test_get_accumulated_dist():
    dist_per_step = np.array([1, 4, 5, 7])
    dist_acc_test = np.array([1, 5, 10, 17])
    dist = graphics.get_accumulated_dist(dist_per_step)

    assert np.array_equal(dist_acc_test, dist)


def test_get_hist_values_from_boundaries():
    bin_boundaries = np.array([1, 4, 5, 7])
    bin_content_unnormalised = np.array([1, 3, 5])
    bin_content_normalised_test = np.array([1 / 3, 3, 5 / 2])
    bin_centres_test = np.array([2.5, 4.5, 6])
    bin_widths_test = np.array([3, 1, 2])

    hist_values = graphics.get_hist_values_from_boundaries(bin_boundaries, bin_content_unnormalised)

    assert np.array_equal(bin_centres_test, hist_values['bin_centres'])
    assert np.array_equal(bin_widths_test, hist_values['bin_widths'])
    assert np.array_equal(bin_content_normalised_test, hist_values['bin_content'])


def test_get_hist_values_from_widths():
    bin_widths = np.array([3, 1, 2]) * u.meter
    bin_content_unnormalised = np.array([1, 3, 5]) * u.kg
    bin_content_normalised_test = np.array([1 / 3, 3, 5 / 2]) * u.kg/u.meter
    bin_centres_test = np.array([1.5, 3.5, 5]) * u.meter

    hist_values = graphics.get_hist_values_from_widths(bin_widths, bin_content_unnormalised, "fuel")

    assert np.array_equal(bin_centres_test, hist_values['bin_centres'])
    assert np.array_equal(bin_content_normalised_test, hist_values['bin_contents'])
