import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from astropy import units as u

import WeatherRoutingTool.utils.unit_conversion as unit
from WeatherRoutingTool.utils.maps import Map


def test_get_angle_bins_2greater360():
    min_alpha = 380 * u.degree
    max_alpha = 400 * u.degree
    spacing = 21

    result = unit.get_angle_bins(min_alpha, max_alpha, spacing)

    assert result[0] == 20 * u.degree
    assert result[result.shape[0] - 1] == 40 * u.degree
    assert (result[1] - result[0]) == 1 * u.degree


def test_get_angle_bins_maxgreater360():
    min_alpha = 260 * u.degree
    max_alpha = 400 * u.degree
    spacing = 141

    result = unit.get_angle_bins(min_alpha, max_alpha, spacing)

    assert result[0] == 260 * u.degree
    assert result[result.shape[0] - 1] == 40 * u.degree
    assert (result[1] - result[0]) == 1 * u.degree


def test_get_angle_bins_2smaller0():
    min_alpha = -40 * u.degree
    max_alpha = -10 * u.degree
    spacing = 31

    result = unit.get_angle_bins(min_alpha, max_alpha, spacing)

    assert result[0] == 320 * u.degree
    assert result[result.shape[0] - 1] == 350 * u.degree
    assert (result[1] - result[0]) == 1 * u.degree


def test_get_angle_bins_minsmaller0():
    min_alpha = -40 * u.degree
    max_alpha = 20 * u.degree
    spacing = 61

    result = unit.get_angle_bins(min_alpha, max_alpha, spacing)

    print('result: ', result)

    assert result[0] == 320 * u.degree
    assert result[result.shape[0] - 1] == 20 * u.degree
    assert (result[1] - result[0]) == 1 * u.degree


def test_downsample_dataframe():
    time = np.array(
        [datetime(2022, 12, 19), datetime(2022, 12, 19) + timedelta(days=1), datetime(2022, 12, 19) + timedelta(days=2),
         datetime(2022, 12, 19) + timedelta(days=3), datetime(2022, 12, 19) + timedelta(days=4),
         datetime(2022, 12, 19) + timedelta(days=5), datetime(2022, 12, 19) + timedelta(days=6)])
    var_1 = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3])
    var_2 = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    var_1_test = np.array([0.5, 2., 3.])
    var_2_test = np.array([0.1, 0.4, 0.6])

    data = {'time': time, 'var_1': var_1, 'var_2': var_2}
    df = pd.DataFrame(data)

    df_resampled = unit.downsample_dataframe(df, 3)
    var_1_returned = df_resampled['var_1'].values
    var_2_returned = df_resampled['var_2'].values

    assert np.allclose(var_1_test, var_1_returned, 0.00001)
    assert np.allclose(var_2_test, var_2_returned, 0.00001)


def test_mps_to_knots_zero():
    assert unit.mps_to_knots(0) == 0


def test_mps_to_knots_positive():
    assert np.isclose(unit.mps_to_knots(1), 1.94384)


def test_mps_to_knots_negative():
    assert np.isclose(unit.mps_to_knots(-10), -19.4384)


def test_mps_to_knots_large_number():
    assert np.isclose(unit.mps_to_knots(1000), 1943.84)


def test_knots_to_mps_zero():
    assert unit.knots_to_mps(0) == 0


def test_knots_to_mps_positive():
    assert np.isclose(unit.knots_to_mps(1), 0.514444)


def test_knots_to_mps_negative():
    assert np.isclose(unit.knots_to_mps(-10), -5.14444)


def test_knots_to_mps_large_number():
    assert np.isclose(unit.knots_to_mps(1000), 514.444)


def test_degree_to_pmpi_zero():
    deg = np.array([0.0]) * u.degree
    rad = unit.degree_to_pmpi(deg)
    assert np.isclose(rad.value, 0.0)


def test_degree_to_pmpi_positive():
    deg = np.array([90.0]) * u.degree
    rad = unit.degree_to_pmpi(deg)
    assert np.isclose(rad.value, np.pi / 2)


def test_degree_to_pmpi_negative():
    deg = np.array([-90.0]) * u.degree
    rad = unit.degree_to_pmpi(deg)
    assert np.isclose(rad.value, -np.pi / 2)


def test_degree_to_pmpi_over_180():
    deg = np.array([270.0]) * u.degree
    rad = unit.degree_to_pmpi(deg)
    assert np.isclose(rad.value, -np.pi / 2)


def test_degree_to_pmpi_over_360():
    deg = np.array([450.0]) * u.degree
    rad = unit.degree_to_pmpi(deg)
    assert np.isclose(rad.value, np.pi / 2)


@pytest.mark.parametrize("input,output", [
    ([-5, -5, -2, -1], [-6, -6, -1, 0]),  # all negative
    ([-5, -1, 5, 1], [-6, -2, 6, 2]),  # min negative, max positive
    ([1, 5, 5, 5], [0, 4, 6, 6]),  # all positive
    ([0, -5, 5, 0], [-1, -6, 6, 1])  # min and max are 0
])
def test_return_widened_map(input, output):
    map = Map(input[0], input[1], input[2], input[3])
    widened_map = map.get_widened_map(1)
    assert widened_map.lat1 == output[0]
    assert widened_map.lon1 == output[1]
    assert widened_map.lat2 == output[2]
    assert widened_map.lon2 == output[3]
