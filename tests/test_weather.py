import math

import pytest

from WeatherRoutingTool.weather import WeatherCond

SQRT2_2 = math.sqrt(2) / 2


@pytest.mark.parametrize("u,v,theta_res", [(-1, -1, 45), (1, -1, 315), (1, 1, 225), (-1, 1, 135)])
def test_theta_from_uv(u, v, theta_res):
    theta_test = WeatherCond.get_theta_from_uv(u, v)

    assert theta_test == pytest.approx(theta_res)


@pytest.mark.parametrize("theta,windspeed,u_res", [
    (0, 1, 0.0),
    (90, 1, -1.0),
    (180, 1, 0.0),
    (270, 1, 1.0),
    (45, 1, -SQRT2_2),
    (135, 1, -SQRT2_2),
    (225, 1, SQRT2_2),
    (315, 1, SQRT2_2),
])
def test_get_u(theta, windspeed, u_res):
    u_test = WeatherCond.get_u(theta, windspeed)

    assert u_test == pytest.approx(u_res)


@pytest.mark.parametrize("theta,windspeed,v_res", [
    (0, 1, -1.0),
    (90, 1, 0.0),
    (180, 1, 1.0),
    (270, 1, 0.0),
    (45, 1, -SQRT2_2),
    (135, 1, SQRT2_2),
    (225, 1, SQRT2_2),
    (315, 1, -SQRT2_2),
])
def test_get_v(theta, windspeed, v_res):
    v_test = WeatherCond.get_v(theta, windspeed)

    assert v_test == pytest.approx(v_res)


@pytest.mark.parametrize("theta,windspeed", [
    (0, 1), (45, 1), (90, 1), (135, 1),
    (180, 1), (225, 1), (270, 1), (315, 1),
    (60, 5), (120, 0.5),
])
def test_uv_magnitude_equals_windspeed(theta, windspeed):
    u = WeatherCond.get_u(theta, windspeed)
    v = WeatherCond.get_v(theta, windspeed)

    assert math.sqrt(u**2 + v**2) == pytest.approx(abs(windspeed))


@pytest.mark.parametrize("theta,windspeed", [
    (0, 1), (45, 1), (90, 1), (135, 1),
    (180, 1), (225, 1), (270, 1), (315, 1),
    (60, 3),
])
def test_uv_roundtrip_recovers_theta(theta, windspeed):
    u = WeatherCond.get_u(theta, windspeed)
    v = WeatherCond.get_v(theta, windspeed)

    assert WeatherCond.get_theta_from_uv(u, v) == pytest.approx(theta)
