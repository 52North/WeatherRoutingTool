import pytest

from WeatherRoutingTool.weather import WeatherCond


@pytest.mark.parametrize("u,v,theta_res", [(-1, -1, 45), (1, -1, 315), (1, 1, 225), (-1, 1, 135)])
def test_theta_from_uv(u, v, theta_res):
    theta_test = WeatherCond.get_theta_from_uv(u, v)

    assert theta_test == pytest.approx(theta_res)


@pytest.mark.parametrize("theta,windspeed,u_res", [
    (45, 1, -0.7071067811865476),
    (315, 1, 0.7071067811865476),
    (225, 1, 0.7071067811865476),
    (135, 1, -0.7071067811865476),
    # cardinal directions
    (0, 1, 0.0),
    (90, 1, -1.0),
    (180, 1, 0.0),
    (270, 1, 1.0),
    # windspeed scaling
    (45, 5, -3.5355339059327378),
    # zero wind
    (45, 0, 0.0),
    # full circle should equal 0 degrees
    (360, 1, 0.0),
])
def test_get_u(theta, windspeed, u_res):
    u_test = WeatherCond.get_u(theta, windspeed)

    assert u_test == pytest.approx(u_res, abs=1e-10)


@pytest.mark.parametrize("theta,windspeed,v_res", [
    (45, 1, -0.7071067811865476),
    (315, 1, -0.7071067811865476),
    (225, 1, 0.7071067811865476),
    (135, 1, 0.7071067811865476),
    # cardinal directions
    (0, 1, -1.0),
    (90, 1, 0.0),
    (180, 1, 1.0),
    (270, 1, 0.0),
    # windspeed scaling
    (45, 5, -3.5355339059327378),
    # zero wind
    (45, 0, 0.0),
    # full circle
    (360, 1, -1.0),
])
def test_get_v(theta, windspeed, v_res):
    v_test = WeatherCond.get_v(theta, windspeed)

    assert v_test == pytest.approx(v_res, abs=1e-10)


@pytest.mark.parametrize("theta,windspeed", [
    (45, 1),
    (135, 1),
    (225, 1),
    (315, 1),
    (45, 10),
    (200, 3.5),
])
def test_uv_roundtrip(theta, windspeed):
    """get_u/get_v -> get_theta_from_uv should return the original angle"""
    u = WeatherCond.get_u(theta, windspeed)
    v = WeatherCond.get_v(theta, windspeed)
    theta_back = WeatherCond.get_theta_from_uv(u, v)

    assert theta_back == pytest.approx(theta, abs=1e-6)
