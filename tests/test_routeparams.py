import os
from datetime import datetime, timedelta

from geovectorslib import geod
import numpy as np
import pytest
from astropy import units as u

import tests.basic_test_func as basic_test_func
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.ship.shipparams import ShipParams


def _make_test_route():
    lats = np.array([40.0, 50.0, 60.0, 70.0])
    lons = np.array([4.0, 5.0, 6.0, 7.0])
    start_times = np.array([
        datetime(2022, 12, 19) + timedelta(hours=h) for h in range(4)
    ])
    sp = ShipParams.set_default_array_1D(3)
    sp.fuel_rate = np.array([1.12, 1.13, 1.15]) * u.kg / u.s
    sp.rpm = np.array([10.0, 11.0, 12.0]) / u.minute
    sp.speed = np.array([6.0, 6.0, 6.0]) * u.m / u.s
    sp.power = np.array([1000.0, 2000.0, 4000.0]) * u.Watt

    return RouteParams(
        count=2,
        start=(lats[0], lons[0]),
        finish=(lats[-1], lons[-1]),
        gcr=None,
        route_type='test',
        time=timedelta(hours=3),
        lats_per_step=lats,
        lons_per_step=lons,
        course_per_step=np.array([10.0, 20.0, 30.0]) * u.degree,
        dists_per_step=np.array([100.0, 200.0, 150.0]) * u.meter,
        starttime_per_step=start_times,
        ship_params_per_step=sp,
    )


def test_routeparams_equality(tmp_path):
    route = _make_test_route()
    equivalent_route = _make_test_route()

    assert route == equivalent_route
    assert not (route != equivalent_route)

    # Compatible units should still compare equal (power: W vs kW)
    equivalent_route.ship_params_per_step.power = (
        np.array([1.0, 2.0, 4.0]) * u.kW
    )
    assert route == equivalent_route

    # Equality of routes loaded from GeoJSON (isolated from shared fixture)
    route_file = tmp_path / 'route.json'
    route.write_to_geojson(str(route_file))
    assert RouteParams.from_file(str(route_file)) == RouteParams.from_file(str(route_file))


@pytest.mark.parametrize('mutate', [
    lambda r: setattr(r.ship_params_per_step, 'fuel_rate', np.array([9.9, 1.13, 1.15]) * u.kg / u.s),
    lambda r: setattr(r.ship_params_per_step, 'fuel_rate', np.array([1.12, 1.13, 1.15]) * u.meter),
    lambda r: setattr(r.ship_params_per_step, 'rpm', np.array([99.0, 11.0, 12.0]) / u.minute),
    lambda r: setattr(r.ship_params_per_step, 'speed', np.array([9.0, 6.0, 6.0]) * u.m / u.s),
    lambda r: setattr(r.ship_params_per_step, 'fuel_type', 'MDO'),
    lambda r: setattr(r, 'count', 99),
    lambda r: setattr(r, 'route_type', 'other'),
    lambda r: setattr(r, 'lats_per_step', np.array([41.0, 50.0, 60.0, 70.0])),
    lambda r: setattr(r, 'starttime_per_step', r.starttime_per_step + timedelta(hours=1)),
])
def test_routeparams_inequality(mutate):
    route = _make_test_route()
    other = _make_test_route()
    mutate(other)

    assert route != other
    assert not (route == other)


def test_routeparams_equality_non_route_type():
    route = _make_test_route()

    assert route != None  # noqa: E711
    assert not (route == None)  # noqa: E711
    assert route != 'not a route'
    assert not (route == 'not a route')


def test_get_acc_variables():
    lats = np.array([40, 50, 60, 70])
    lons = np.array([4, 5, 6, 7])
    fuel_rate = np.array([1.12, 1.13, 1.15]) * u.kg / u.s
    dist = np.array([100, 200, 150]) * u.meter
    start_time = np.array([datetime(2022, 12, 19),
                           datetime(2022, 12, 19) + timedelta(hours=1),
                           datetime(2022, 12, 19) + timedelta(hours=2),
                           datetime(2022, 12, 19) + timedelta(hours=3)
                           ])
    dummy = np.array([0, 0, 0, 0])

    sp = ShipParams(
        fuel_rate=fuel_rate,
        power=dummy * u.Watt,
        rpm=dummy * u.Hz,
        speed=dummy * u.m / u.s,
        r_calm=dummy * u.N,
        r_wind=dummy * u.N,
        r_waves=dummy * u.N,
        r_shallow=dummy * u.N,
        r_roughness=dummy * u.N,
        wave_height=dummy * u.m,
        wave_direction=dummy * u.rad,
        wave_period=dummy * u.second,
        u_currents=dummy * u.m / u.s,
        v_currents=dummy * u.m / u.s,
        u_wind_speed=dummy * u.m / u.s,
        v_wind_speed=dummy * u.m / u.s,
        pressure=dummy * u.kg / u.meter / u.second ** 2,
        air_temperature=dummy * u.deg_C,
        salinity=dummy * u.dimensionless_unscaled,
        water_temperature=dummy * u.deg_C,
        message=dummy,
        status=dummy

    )

    rp = RouteParams(
        count=2,
        start=(lons[0], lats[0]),
        finish=(lons[-1], lats[-1]),
        gcr=None,
        route_type='test',
        time=dummy,
        lats_per_step=lats,
        lons_per_step=lons,
        course_per_step=dummy,
        dists_per_step=dist,
        starttime_per_step=start_time,
        ship_params_per_step=sp
    )
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'data/min_fuel_route_test.json')

    rp.write_to_geojson(filename)
    rp_test = RouteParams.from_file(filename)

    test_fuel = (1.12 + 1.13 + 1.15) * 3600 * u.kg
    test_dist = 3344981 * u.meter
    test_time = timedelta(hours=3)

    assert test_fuel == rp_test.get_full_fuel()
    assert np.allclose(test_dist, rp_test.get_full_dist())
    assert test_time == rp_test.get_full_travel_time()


'''
    Test whether parameters in `RouteParams.get_per_waypoint_coords` are calculated correctly.
'''


def test_get_waypoint_coords():
    bs = 6 * u.meter / u.second
    start_time = datetime.strptime("2023-07-20T10:00Z", '%Y-%m-%dT%H:%MZ')
    route_lats = np.array([54.9, 54.7, 54.5, 54.2])
    route_lons = np.array([13.2, 13.4, 13.7, 13.9])

    start_lats_test = np.array([54.9, 54.7, 54.5])
    start_lons_test = np.array([13.2, 13.4, 13.7])
    dists_test = np.full(route_lats.shape[0] - 1, -99.)
    start_times_test = np.full(route_lats.shape[0] - 1, start_time)
    start_times_test[0] = start_time
    travel_times_test = np.full(route_lats.shape[0] - 1, timedelta(seconds=0.))

    for ipoint in range(3):
        start_lat = route_lats[ipoint]
        start_lon = route_lons[ipoint]
        end_lat = route_lats[ipoint + 1]
        end_lon = route_lons[ipoint + 1]

        dists_test[ipoint] = geod.inverse(
            [start_lat],
            [start_lon],
            [end_lat],
            [end_lon]
        )['s12'][0]
        travel_times_test[ipoint] = (dists_test[ipoint] * u.meter / bs).value
        if ipoint < 2:
            start_times_test[ipoint + 1] = start_times_test[ipoint] + timedelta(seconds=travel_times_test[ipoint])

    waypoint_dict = RouteParams.get_per_waypoint_coords(route_lons, route_lats, start_time, bs)

    assert np.all(start_lats_test == waypoint_dict['start_lats'])
    assert np.all(start_lons_test == waypoint_dict['start_lons'])
    assert np.all(dists_test == waypoint_dict['dist'].value)
    assert np.all(start_times_test == waypoint_dict['start_times'])
    assert np.all(travel_times_test == waypoint_dict['travel_times'].value)


'''
    Test whether parameters in `RouteParams.get_per_waypoint_coords` are calculated correctly for boat speed array.
'''


def test_get_waypoint_coords_bsarray():
    bs = np.array([6, 7, 8]) * u.meter / u.second
    start_time = datetime.strptime("2023-07-20T10:00Z", '%Y-%m-%dT%H:%MZ')
    route_lats = np.array([54.9, 54.7, 54.5, 54.2])
    route_lons = np.array([13.2, 13.4, 13.7, 13.9])

    start_lats_test = np.array([54.9, 54.7, 54.5])
    start_lons_test = np.array([13.2, 13.4, 13.7])
    dists_test = np.full(route_lats.shape[0] - 1, -99.)
    start_times_test = np.full(route_lats.shape[0] - 1, start_time)
    start_times_test[0] = start_time
    travel_times_test = np.full(route_lats.shape[0] - 1, timedelta(seconds=0.))

    for ipoint in range(3):
        start_lat = route_lats[ipoint]
        start_lon = route_lons[ipoint]
        end_lat = route_lats[ipoint + 1]
        end_lon = route_lons[ipoint + 1]

        dists_test[ipoint] = geod.inverse(
            [start_lat],
            [start_lon],
            [end_lat],
            [end_lon]
        )['s12'][0]
        travel_times_test[ipoint] = (dists_test[ipoint] * u.meter / bs[ipoint]).value
        if ipoint < 2:
            start_times_test[ipoint + 1] = start_times_test[ipoint] + timedelta(seconds=travel_times_test[ipoint])

    waypoint_dict = RouteParams.get_per_waypoint_coords(route_lons, route_lats, start_time, bs)

    assert np.all(start_lats_test == waypoint_dict['start_lats'])
    assert np.all(start_lons_test == waypoint_dict['start_lons'])
    assert np.all(dists_test == waypoint_dict['dist'].value)
    assert np.all(start_times_test == waypoint_dict['start_times'])
    assert np.all(travel_times_test == waypoint_dict['travel_times'].value)
