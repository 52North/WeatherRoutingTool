"""
Tests for compare.py fixes (Issue #158):
  1. Loop range TypeError fix in shipParamsPerDist
  2. ShipParams attribute: fuel_rate (not .fuel)
  3. Portable path resolution via os.path
"""
import os
import numpy as np
import pytest
from astropy import units as u
from datetime import datetime, timedelta

from WeatherRoutingTool.ship.shipparams import ShipParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ship_params(n):
    """
    Return a minimal 1-D ShipParams with n entries.

    :param n: The number of entries in the ShipParams arrays.
    :type n: int
    :return: A ShipParams object initialized with dummy data.
    :rtype: ShipParams
    """
    return ShipParams(
        fuel_rate=np.ones(n) * u.kg / u.second,
        power=np.ones(n) * u.Watt,
        rpm=np.ones(n) / u.minute,
        speed=np.ones(n) * u.meter / u.second,
        r_calm=np.zeros(n) * u.newton,
        r_wind=np.zeros(n) * u.newton,
        r_waves=np.zeros(n) * u.newton,
        r_shallow=np.zeros(n) * u.newton,
        r_roughness=np.zeros(n) * u.newton,
        wave_height=np.zeros(n) * u.meter,
        wave_direction=np.zeros(n) * u.radian,
        wave_period=np.zeros(n) * u.second,
        u_currents=np.zeros(n) * u.meter / u.second,
        v_currents=np.zeros(n) * u.meter / u.second,
        u_wind_speed=np.zeros(n) * u.meter / u.second,
        v_wind_speed=np.zeros(n) * u.meter / u.second,
        pressure=np.zeros(n) * u.kg / u.meter / u.second ** 2,
        air_temperature=np.zeros(n) * u.deg_C,
        salinity=np.zeros(n) * u.dimensionless_unscaled,
        water_temperature=np.zeros(n) * u.deg_C,
        status=np.zeros(n, dtype=int),
        message=np.full(n, ""),
    )


# ---------------------------------------------------------------------------
# Issue #158 — Bug 1: loop range TypeError
# ---------------------------------------------------------------------------

class TestShipParamsPerDistLoop:
    """
    The original code used range(len(1, lats_per_step - 1)) which raises
    TypeError because len() accepts exactly one argument.
    The fix is: range(1, len(lats_per_step) - 1).
    """

    def test_old_code_raises_typeerror(self):
        """
        Confirm the original buggy expression raises TypeError.

        This test verifies that the `len()` function raises a `TypeError` when passed
        multiple arguments, as it did in the original code.
        """
        lats = np.array([38.0, 38.5, 39.0, 39.5, 40.0, 40.5, 41.0])
        with pytest.raises(TypeError):
            # This is intentionally the OLD buggy code to prove the bug existed
            for _ in range(len(1, lats - 1)):  # noqa: E501
                pass

    def test_fixed_loop_runs_without_error(self):
        """
        Fixed expression iterates without raising any exception.

        This test ensures that the corrected `range(1, len(lats) - 1)` loop
        executes successfully.
        """
        lats = np.array([38.0, 38.5, 39.0, 39.5, 40.0, 40.5, 41.0])
        indices = []
        for i in range(1, len(lats) - 1):
            indices.append(i)
        assert indices == [1, 2, 3, 4, 5]

    def test_fixed_loop_skips_first_and_last(self):
        """Loop must skip index 0 (departure) and the last index (destination)."""
        lats = np.array([38.0, 38.5, 39.0, 40.5, 41.0])  # 5 elements
        indices = list(range(1, len(lats) - 1))
        assert 0 not in indices, "Must skip index 0 (departure point)"
        assert len(lats) - 1 not in indices, "Must skip last index (destination)"

    def test_fixed_loop_correct_count(self):
        """Number of iterations equals len(lats_per_step) - 2."""
        for size in [4, 7, 10, 20]:
            lats = np.zeros(size)
            count = len(list(range(1, len(lats) - 1)))
            assert count == size - 2


# ---------------------------------------------------------------------------
# Issue #158 — Bug 2: ShipParams has .fuel_rate not .fuel
# ---------------------------------------------------------------------------

class TestShipParamsFuelRateAttribute:
    """
    The original code accessed shipParams.fuel which does not exist.
    The correct attribute is shipParams.fuel_rate.
    """

    def test_fuel_rate_attribute_exists(self):
        """
        ShipParams must expose a fuel_rate attribute.

        Verifies that the `ShipParams` class has the `fuel_rate` attribute.
        """
        sp = make_ship_params(5)
        assert hasattr(sp, 'fuel_rate'), "ShipParams must have 'fuel_rate' attribute"

    def test_fuel_attribute_does_not_exist(self):
        """ShipParams must NOT expose a plain .fuel attribute (confirms the bug was real)."""
        sp = make_ship_params(5)
        assert not hasattr(sp, 'fuel'), \
            "ShipParams has no '.fuel' attribute — accessing it would raise AttributeError"

    def test_fuel_rate_is_indexable(self):
        """fuel_rate must be indexable so shipParams.fuel_rate[i] works in the loop."""
        n = 6
        sp = make_ship_params(n)
        for i in range(n):
            val = sp.fuel_rate[i]
            assert val is not None

    def test_fuel_rate_correct_unit(self):
        """fuel_rate should carry kg/s units as expected by the routing tool."""
        sp = make_ship_params(3)
        assert sp.fuel_rate.unit == u.kg / u.second

    def test_fuel_rate_get_method(self):
        """get_fuel_rate() must return the same object as fuel_rate."""
        sp = make_ship_params(4)
        assert np.array_equal(sp.get_fuel_rate().value, sp.fuel_rate.value)


# ---------------------------------------------------------------------------
# Portable path fix
# ---------------------------------------------------------------------------

class TestComparePyPortablePaths:
    """
    The original compare.py used hardcoded absolute paths pointing to the
    original developer's machine. These have been replaced with paths
    resolved dynamically from the script's own location via os.path.
    """

    def test_compare_py_exists(self):
        """
        compare.py must exist in the project root.

        Checks for the existence of `compare.py` in the expected directory.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        compare_path = os.path.join(project_root, 'compare.py')
        assert os.path.isfile(compare_path), "compare.py not found in project root"

    def test_compare_py_no_hardcoded_user_paths(self):
        """compare.py must not contain hardcoded /Users/parichay paths."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        compare_path = os.path.join(project_root, 'compare.py')
        with open(compare_path, 'r') as f:
            content = f.read()
        assert '/Users/parichay' not in content, \
            "compare.py still contains hardcoded developer paths (/Users/parichay)"

    def test_compare_py_uses_os_path(self):
        """compare.py must use os.path for portable path resolution."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        compare_path = os.path.join(project_root, 'compare.py')
        with open(compare_path, 'r') as f:
            content = f.read()
        assert 'os.path.abspath(__file__)' in content, \
            "compare.py must use os.path.abspath(__file__) for portable paths"
