import json
import pytest
from pathlib import Path
from WeatherRoutingTool.config import Config


def load_example_config():
    config_path = Path(__file__).parent / "config.tests.json"
    with config_path.open("r") as f:
        return json.load(f), config_path


def test_assign_config_from_json():
    _, config_path = load_example_config()
    config = Config.assign_config(path=config_path, init_mode="from_json")

    assert isinstance(config, Config)
    assert config.CONFIG_PATH == config_path
    assert config.DEPARTURE_TIME is not None
    assert config.ISOCHRONE_PRUNE_SYMMETRY_AXIS == 'gcr'


def test_assign_config_from_dict():
    config_data, _ = load_example_config()
    config = Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert isinstance(config, Config)
    assert config.DEPARTURE_TIME is not None
    assert config.ISOCHRONE_PRUNE_SYMMETRY_AXIS == 'gcr'


def test_invalid_time_raises_error():
    config_data, _ = load_example_config()
    config_data["DEPARTURE_TIME"] = "2023-11-11T1111Z"
    with pytest.raises(ValueError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert "'DEPARTURE_TIME' must be in format YYYY-MM-DDTHH:MMZ" in str(excinfo.value)


def test_invalid_path_raises_error(tmp_path):
    config_data, _ = load_example_config()
    config_data["ROUTE_PATH"] = str(tmp_path / "nonexistent.nc")

    with pytest.raises(ValueError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert "Path doesn't exist" in str(excinfo.value)


def test_speedy_boat_validation_fails():
    config_data, _ = load_example_config()

    # Set inconsistent values
    config_data["BOAT_TYPE"] = "speedy_isobased"
    config_data["ALGORITHM_TYPE"] = "isofuel"

    with pytest.raises(ValueError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert "If 'BOAT_TYPE' or 'ALGORITHM_TYPE' is 'speedy_isobased'" in str(excinfo.value)


def test_invalid_route_raises_error():
    config_data, _ = load_example_config()
    config_data["DEFAULT_ROUTE"] = [54.15, 13.15, 54.56, 200]

    with pytest.raises(ValueError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert "lon_end must be between -180 and 180" in str(excinfo.value)


def test_negative_delta_fuel_raises_error():
    config_data, _ = load_example_config()
    config_data["DELTA_FUEL"] = -100
    with pytest.raises(ValueError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert "'DELTA_FUEL' must be greater than zero" in str(excinfo.value)


def test_non_even_router_hdgs_segments_raises_error():
    config_data, _ = load_example_config()
    config_data["ROUTER_HDGS_SEGMENTS"] = 31
    with pytest.raises(ValueError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert "'ROUTER_HDGS_SEGMENTS' must be a positive even integer" in str(excinfo.value)


def test_route_map_compatibility():
    config_data, _ = load_example_config()
    config_data["DEFAULT_ROUTE"] = [36.192, 13.392, 41.349, 17.188]
    with pytest.raises(ValueError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert "is outside the defined map bounds" in str(excinfo.value)


def test_weather_start_time_compatibility():
    config_data, _ = load_example_config()
    config_data["DEPARTURE_TIME"] = "2022-04-01T11:11Z"
    with pytest.raises(ValueError) as excinfo:
        Config.assign_config(init_mode="from_dict", config_dict=config_data)

    assert "Weather data does not cover the full routing time range." in str(excinfo.value)
