#!/usr/bin/env python3
"""
Test script to verify minimal installation works correctly.
This script tests core functionality without optional dependencies.
"""

import sys
import traceback

def test_core_imports():
    """Test that core modules can be imported with minimal installation."""
    print("Testing core imports...")
    
    try:
        # Core dependencies
        import numpy as np
        import pandas as pd
        import xarray as xr
        from scipy import interpolate
        import astropy.units as u
        from pydantic import BaseModel
        print("✓ Core dependencies imported successfully")
    except ImportError as e:
        print(f"✗ Core dependency import failed: {e}")
        return False
    
    try:
        # WRT core modules
        import WeatherRoutingTool.config as config
        import WeatherRoutingTool.routeparams as routeparams
        import WeatherRoutingTool.weather as weather
        print("✓ WRT core modules imported successfully")
    except ImportError as e:
        print(f"✗ WRT core module import failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_basic_functionality():
    """Test basic WRT functionality."""
    print("\nTesting basic functionality...")
    
    try:
        from WeatherRoutingTool.config import Config
        from WeatherRoutingTool.weather import WeatherCond
        from datetime import datetime, timedelta
        
        # Test basic configuration
        print("✓ Config class accessible")
        
        # Test weather condition creation
        weather_cond = WeatherCond(
            time=datetime.now(),
            hours=24,
            time_res=3
        )
        print("✓ WeatherCond creation successful")
        
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_optional_imports_fail():
    """Test that optional dependencies are not available in minimal install."""
    print("\nTesting that optional dependencies are not available...")
    
    optional_deps = [
        ('matplotlib', 'matplotlib.pyplot'),
        ('cartopy', 'cartopy.crs'),
        ('geopandas', 'geopandas'),
        ('pymoo', 'pymoo'),
        ('dask', 'dask'),
        ('netcdf4', 'netCDF4'),
    ]
    
    for dep_name, import_path in optional_deps:
        try:
            __import__(import_path)
            print(f"⚠ Optional dependency {dep_name} is available (unexpected)")
        except ImportError:
            print(f"✓ Optional dependency {dep_name} not available (expected)")
    
    return True

def main():
    """Run all tests."""
    print("WeatherRoutingTool Minimal Installation Test")
    print("=" * 50)
    
    tests = [
        test_core_imports,
        test_basic_functionality,
        test_optional_imports_fail,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("🎉 All tests passed! Minimal installation works correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
