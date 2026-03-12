"""
Shared pytest fixtures for the WeatherRoutingTool test suite.

Fixtures are organised by scope:
- ``session``  : expensive objects loaded once per test run (ship/algorithm objects
                 backed by file I/O or heavy initialisation)
- ``function`` : lightweight in-memory objects that are recreated for every test
                 to prevent inter-test state leakage
"""

import os
from pathlib import Path

import pytest

import tests.basic_test_func as basic_test_func
from WeatherRoutingTool.algorithms.isobased import IsoBased
from WeatherRoutingTool.algorithms.isofuel import IsoFuel
from WeatherRoutingTool.config import Config
from WeatherRoutingTool.ship.direct_power_boat import DirectPowerBoat
from WeatherRoutingTool.ship.ship_config import ShipConfig


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_data_dir() -> str:
    """Absolute path to the ``tests/data/`` directory."""
    return os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="session")
def tests_config_path() -> Path:
    """Path to the main test configuration file ``tests/config.tests.json``."""
    return Path(__file__).parent / "config.tests.json"


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def wrt_config(tests_config_path) -> Config:
    """
    ``Config`` object loaded from ``config.tests.json``.

    Session-scoped because the config file never changes during a test run.
    """
    return Config.assign_config(path=tests_config_path)


@pytest.fixture(scope="session")
def ship_config(tests_config_path) -> ShipConfig:
    """
    ``ShipConfig`` object loaded from ``config.tests.json``.

    Session-scoped because the config file never changes during a test run.
    """
    return ShipConfig.assign_config(path=tests_config_path)


# ---------------------------------------------------------------------------
# Ship fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def simpleship() -> DirectPowerBoat:
    """
    ``DirectPowerBoat`` whose ship geometry is **auto-calculated** from the
    mandatory parameters only (hs1, ls1, … are approximated internally).

    Use this fixture whenever the test only cares about power / wind
    calculations and does not need to verify the geometry itself.
    """
    return basic_test_func.create_dummy_Direct_Power_Ship("simpleship")


@pytest.fixture(scope="session")
def manualship() -> DirectPowerBoat:
    """
    ``DirectPowerBoat`` with all ship geometry parameters supplied manually.

    Use this fixture when the test needs to verify Fujiwara wind-resistance
    coefficients or geometry-dependent behaviour.
    """
    return basic_test_func.create_dummy_Direct_Power_Ship("manualship")


@pytest.fixture
def base_ship_config_dict() -> dict:
    """
    Minimal valid ship-configuration dictionary (simpleship parameters).

    Function-scoped so that each test receives a fresh copy it can mutate
    freely without affecting other tests.
    """
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
    """
    ``DirectPowerBoat`` constructed from a plain dictionary (no JSON file
    required).  Useful for testing ``ShipConfig.assign_config`` paths and
    for tests that need to vary individual config values.
    """
    ship_config = ShipConfig.assign_config(
        init_mode="from_dict", config_dict=base_ship_config_dict
    )
    boat = DirectPowerBoat(ship_config)
    boat.load_data()
    return boat


# ---------------------------------------------------------------------------
# Routing algorithm fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def isofuel_algorithm() -> IsoFuel:
    """
    ``IsoFuel`` routing algorithm object initialised from ``config.tests.json``.

    Session-scoped because algorithm construction is expensive and the object
    is treated as read-only by the tests that use it.
    """
    return basic_test_func.create_dummy_IsoFuel_object()


@pytest.fixture(scope="session")
def isobased_algorithm() -> IsoBased:
    """
    ``IsoBased`` routing algorithm object initialised from ``config.tests.json``.

    Session-scoped for the same reason as ``isofuel_algorithm``.
    """
    return basic_test_func.create_dummy_IsoBased_object()
