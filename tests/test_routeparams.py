import os
from datetime import datetime, timedelta

import numpy as np
from astropy import units as u

import tests.basic_test_func as basic_test_func
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.ship.shipparams import ShipParams


def test_get_acc_variables():
    lats = np.array([40, 50, 60, 70])
    lons = np.array([4, 5, 6, 7])
    fuel_rate = np.array([1.12, 1.13, 1.15]) * u.kg/u.s
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
        speed=dummy * u.m/u.s,
        r_calm=dummy * u.N,
        r_wind=dummy * u.N,
        r_waves=dummy * u.N,
        r_shallow=dummy * u.N,
        r_roughness=dummy * u.N,
        wave_height=dummy * u.m,
        wave_direction=dummy * u.rad,
        wave_period=dummy * u.second,
        u_currents=dummy * u.m/u.s,
        v_currents=dummy * u.m/u.s,
        u_wind_speed=dummy * u.m/u.s,
        v_wind_speed=dummy * u.m/u.s,
        pressure=dummy * u.kg/u.meter/u.second**2,
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
