import pytest

from WeatherRoutingTool.weather import WeatherCond

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("u,v,theta_res", [(-1, -1, 45), (1, -1, 315), (1, 1, 225), (-1, 1, 135)])
def test_theta_from_uv(u, v, theta_res):
    theta_test = WeatherCond.get_theta_from_uv(u, v)

    assert theta_test == pytest.approx(theta_res)


@pytest.mark.parametrize("theta,windspeed,u_res", [(45, 1, -1), (315, 1, 1), (225, 1, 1), (135, 1, -1)])
def get_u(theta, windspeed):
    u_test = WeatherCond.get_u(theta, windspeed)

    assert u_test == pytest.approx(u_test)


@pytest.mark.parametrize("theta,windspeed,u_res", [(45, 1, -1), (315, 1, -1), (225, 1, 1), (135, 1, 1)])
def get_v(theta, windspeed):
    v_test = WeatherCond.get_v(theta, windspeed)

    assert v_test == pytest.approx(v_test)
