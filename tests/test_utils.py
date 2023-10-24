import pytest

import WeatherRoutingTool.utils.unit_conversion as unit

def test_get_angle_bins_2greater360():
    min_alpha=380
    max_alpha=400
    spacing = 21

    result = unit.get_angle_bins(min_alpha, max_alpha, spacing)

    assert result[0] == 20
    assert result[result.shape[0]-1] == 40
    assert (result[1]-result[0]) == 1

def test_get_angle_bins_maxgreater360():
    min_alpha=260
    max_alpha=400
    spacing = 141

    result = unit.get_angle_bins(min_alpha, max_alpha, spacing)

    assert result[0] == 260
    assert result[result.shape[0] - 1] == 40
    assert (result[1] - result[0]) == 1

def test_get_angle_bins_2smaller0():
    min_alpha=-40
    max_alpha=-10
    spacing = 31

    result = unit.get_angle_bins(min_alpha, max_alpha, spacing)

    assert result[0] == 320
    assert result[result.shape[0] - 1] == 350
    assert (result[1] - result[0]) == 1

def test_get_angle_bins_minsmaller0():
    min_alpha=-40
    max_alpha=20
    spacing = 61

    result = unit.get_angle_bins(min_alpha, max_alpha, spacing)

    print('result: ', result)

    assert result[0] == 320
    assert result[result.shape[0] - 1] == 20
    assert (result[1] - result[0]) == 1
