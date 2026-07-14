import pytest

from WeatherRoutingTool.utils.formatting import (
    get_bbox_from_string,
    get_log_step,
    get_line_string,
    get_point_from_string,
)


class TestGetLineString:
    def test_returns_string(self):
        assert isinstance(get_line_string(), str)

    def test_not_empty(self):
        assert len(get_line_string()) > 0


class TestGetLogStep:
    def test_default_indent(self):
        result = get_log_step("hello")
        assert result.endswith("hello")
        assert result.startswith(" ")

    def test_higher_indent_is_longer(self):
        result0 = get_log_step("msg", istep=0)
        result2 = get_log_step("msg", istep=2)
        assert len(result2) > len(result0)

    def test_content_preserved(self):
        assert "my message" in get_log_step("my message", istep=1)


class TestGetPointFromString:
    def test_basic(self):
        lat, lon = get_point_from_string("53.5,10.0")
        assert lat == 53.5
        assert lon == 10.0

    def test_negative_coords(self):
        lat, lon = get_point_from_string("-33.9,-70.6")
        assert lat == -33.9
        assert lon == -70.6

    def test_returns_floats(self):
        lat, lon = get_point_from_string("0,0")
        assert isinstance(lat, float)
        assert isinstance(lon, float)

    def test_missing_comma_raises(self):
        with pytest.raises((ValueError, TypeError)):
            get_point_from_string("53.5 10.0")


class TestGetBboxFromString:
    def test_basic(self):
        result = get_bbox_from_string("50.0,5.0,55.0,15.0")
        assert result == (50.0, 5.0, 55.0, 15.0)

    def test_negative_sentinel(self):
        result = get_bbox_from_string("-99")
        assert result == (0.0, 0.0, 0.0, 0.0)

    def test_negative_coords(self):
        lat1, lon1, lat2, lon2 = get_bbox_from_string("-10,-20,10,20")
        assert lat1 == -10.0
        assert lon1 == -20.0
        assert lat2 == 10.0
        assert lon2 == 20.0

    def test_returns_floats(self):
        result = get_bbox_from_string("1,2,3,4")
        assert all(isinstance(v, float) for v in result)
