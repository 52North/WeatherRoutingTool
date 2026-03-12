import os
from pathlib import Path

import pytest

import tests.basic_test_func as basic_test_func
from WeatherRoutingTool.algorithms.isobased import IsoBased
from WeatherRoutingTool.algorithms.isofuel import IsoFuel
from WeatherRoutingTool.config import Config
from WeatherRoutingTool.ship.direct_power_boat import DirectPowerBoat
from WeatherRoutingTool.ship.ship_config import ShipConfig


@pytest.fixture(scope="session")
def test_data_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="session")
def tests_config_path() -> Path:
    return Path(__file__).parent / "config.tests.json"


@pytest.fixture(scope="session")
def wrt_config(tests_config_path) -> Config:
    return Config.assign_config(path=tests_config_path)


@pytest.fixture(scope="session")
def ship_config(tests_config_path) -> ShipConfig:
    return ShipConfig.assign_config(path=tests_config_path)


@pytest.fixture(scope="session")
def simpleship() -> DirectPowerBoat:
    return basic_test_func.create_dummy_Direct_Power_Ship("simpleship")


@pytest.fixture(scope="session")
def manualship() -> DirectPowerBoat:
    return basic_test_func.create_dummy_Direct_Power_Ship("manualship")


@pytest.fixture
def base_ship_config_dict() -> dict:
    return {
        "BOAT_BREADTH": 32,
        "BOAT_FUEL_RATE": 167,
        "BOAT_HBR": 30,
        "BOAT_LENGTH": 180,
        "BOAT_SMCR_POWER": 6502,
        "BOAT_SMCR_SPEED": 6,
        "BOAT_SPEED": 6,
        "WEATHER_DATA": "abc",
    }


@pytest.fixture
def simpleship_from_dict(base_ship_config_dict) -> DirectPowerBoat:
    ship_config = ShipConfig.assign_config(
        init_mode="from_dict", config_dict=base_ship_config_dict
    )
    boat = DirectPowerBoat(ship_config)
    boat.load_data()
    return boat


@pytest.fixture(scope="session")
def isofuel_algorithm() -> IsoFuel:
    return basic_test_func.create_dummy_IsoFuel_object()


@pytest.fixture(scope="session")
def isobased_algorithm() -> IsoBased:
    return basic_test_func.create_dummy_IsoBased_object()
