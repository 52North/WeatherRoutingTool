import json
import pytest
from pathlib import Path
from WeatherRoutingTool.ship.ship_config import ShipConfig

mandatory_fields = [
    'BOAT_BREADTH',
    'BOAT_FUEL_RATE',
    'BOAT_HBR',
    'BOAT_LENGTH',
    'BOAT_SMCR_POWER',
    'BOAT_SPEED',
    'WEATHER_DATA'
]


def test_mandatory_fields_present():
    model_fields = ShipConfig.model_fields.keys()

    for field in mandatory_fields:
        assert field in model_fields, f"Field '{field}' not found in ConfigModel"


def load_example_config():
    config_path = Path(__file__).parent / "config.tests.json"
    with config_path.open("r") as f:
        return json.load(f), config_path


def test_assign_config_from_json():
    _, config_path = load_example_config()
    config = ShipConfig.assign_config(path=config_path, init_mode="from_json")

    assert isinstance(config, ShipConfig)


def test_negative_boat_speed_raises_error():
    config_data, _ = load_example_config()
    config_data["BOAT_SPEED"] = -5
    with pytest.raises(ValueError) as excinfo:
        ShipConfig.assign_config(init_mode="from_dict", config_dict=config_data)

    assert "'BOAT_SPEED' must be greater than zero" in str(excinfo.value)


def test_invalid_propulsion_efficiency_raises_error():
    config_data, _ = load_example_config()
    config_data["BOAT_PROPULSION_EFFICIENCY"] = 1.1  # > 1
    with pytest.raises(ValueError) as excinfo:
        ShipConfig.assign_config(init_mode="from_dict", config_dict=config_data)

    assert "'BOAT_PROPULSION_EFFICIENCY' must be between 0 and 1" in str(excinfo.value)
