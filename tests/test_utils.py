import numpy as np
from datetime import datetime, timedelta
from astropy import units as u

import WeatherRoutingTool.utils.unit_conversion as unit
import pandas as pd


def test_get_angle_bins_2greater360():
    min_alpha = 380 * u.degree
    max_alpha = 400 * u.degree
    spacing = 21 * u.degree

    result = unit.get_angle_bins(min_alpha, max_alpha, spacing)

    assert result[0] == 20 * u.degree
    assert result[result.shape[0] - 1] == 40 * u.degree
    assert (result[1] - result[0]) == 1 * u.degree


def test_get_angle_bins_maxgreater360():
    min_alpha = 260 * u.degree
    max_alpha = 400 * u.degree
    spacing = 141 * u.degree

    result = unit.get_angle_bins(min_alpha, max_alpha, spacing)

    assert result[0] == 260 * u.degree
    assert result[result.shape[0] - 1] == 40 * u.degree
    assert (result[1] - result[0]) == 1 * u.degree


def test_get_angle_bins_2smaller0():
    min_alpha = -40 * u.degree
    max_alpha = -10 * u.degree
    spacing = 31 * u.degree

    result = unit.get_angle_bins(min_alpha, max_alpha, spacing)

    assert result[0] == 320 * u.degree
    assert result[result.shape[0] - 1] == 350 * u.degree
    assert (result[1] - result[0]) == 1 * u.degree


def test_get_angle_bins_minsmaller0():
    min_alpha = -40 * u.degree
    max_alpha = 20 * u.degree
    spacing = 61 * u.degree

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
