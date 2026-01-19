import sys
import os
from unittest.mock import MagicMock
import numpy as np
import pytest

# Mock dependencies before imports to avoid MissingModule and Metaclass errors
sys.modules['cartopy'] = MagicMock()
sys.modules['cartopy.crs'] = MagicMock()
sys.modules['cartopy.feature'] = MagicMock()
sys.modules['datacube'] = MagicMock()
sys.modules['geopandas'] = MagicMock()
sys.modules['shapely'] = MagicMock()
sys.modules['shapely.geometry'] = MagicMock()
sys.modules['shapely.strtree'] = MagicMock()
sys.modules['global_land_mask'] = MagicMock()
sys.modules['maridatadownloader'] = MagicMock()
sys.modules['geographiclib'] = MagicMock()
sys.modules['geographiclib.geodesic'] = MagicMock()
sys.modules['skimage'] = MagicMock()
sys.modules['skimage.draw'] = MagicMock()
sys.modules['skimage.graph'] = MagicMock()

# Mock pymoo hierarchy
sys.modules['pymoo'] = MagicMock()
sys.modules['pymoo.algorithms'] = MagicMock()
sys.modules['pymoo.algorithms.moo'] = MagicMock()
sys.modules['pymoo.algorithms.moo.nsga2'] = MagicMock()
sys.modules['pymoo.core'] = MagicMock()

# Define a real class for ElementwiseProblem to avoid MagicMock inheritance issues
class MockElementwiseProblem:
    def __init__(self, n_var=1, n_obj=1, n_constr=1, **kwargs):
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_constr = n_constr

# logic to mock ElementwiseProblem correctly
problem_module = MagicMock()
problem_module.ElementwiseProblem = MockElementwiseProblem
sys.modules['pymoo.core.problem'] = problem_module

sys.modules['pymoo.core.result'] = MagicMock()
sys.modules['pymoo.core.sampling'] = MagicMock()
sys.modules['pymoo.core.crossover'] = MagicMock()
sys.modules['pymoo.core.mutation'] = MagicMock()
sys.modules['pymoo.core.repair'] = MagicMock()
sys.modules['pymoo.core.duplicate'] = MagicMock()
sys.modules['pymoo.optimize'] = MagicMock()
sys.modules['pymoo.termination'] = MagicMock()
sys.modules['pymoo.util'] = MagicMock()
sys.modules['pymoo.util.running_metric'] = MagicMock()

# Mock internal data_utils to avoid metaclass conflict
data_utils_mock = MagicMock()
class MockGridMixin:
    pass
data_utils_mock.GridMixin = MockGridMixin
sys.modules['WeatherRoutingTool.algorithms.data_utils'] = data_utils_mock

# Mock population module entirely to avoid massive dependency chain and metaclass issues
sys.modules['WeatherRoutingTool.algorithms.genetic.population'] = MagicMock()
sys.modules['WeatherRoutingTool.algorithms.genetic.crossover'] = MagicMock()
sys.modules['WeatherRoutingTool.algorithms.genetic.mutation'] = MagicMock()
sys.modules['WeatherRoutingTool.algorithms.genetic.repair'] = MagicMock()

geovectorslib_mock = MagicMock()
sys.modules['geovectorslib'] = geovectorslib_mock

# Setup geod.inverse return value
def mock_inverse(lats1, lons1, lats2, lons2):
    lats1 = np.array(lats1)
    lons1 = np.array(lons1)
    lats2 = np.array(lats2)
    lons2 = np.array(lons2)
    # simple euclidean distance for checking
    dists = np.sqrt((lats1 - lats2)**2 + (lons1 - lons2)**2) * 111000 # very rough degrees to meters
    return {'s12': dists, 'azi1': np.zeros_like(dists)}

geovectorslib_mock.geod.inverse.side_effect = mock_inverse

from WeatherRoutingTool.config import Config
from WeatherRoutingTool.algorithms.genetic.problem import RoutingProblem
from WeatherRoutingTool.algorithms.genetic import Genetic
import tests.basic_test_func as basic_test_func
from astropy import units as u

class MockBoat:
    def get_boat_speed(self):
        class MockSpeed:
             value = 10
        return MockSpeed() 
        
    def get_ship_parameters(self, courses, lats, lons, times):
        class MockParams:
            def get_fuel_rate(self):
                return np.ones(len(courses)) * 100 # Dummy fuel rate
            def get_speed(self):
                # For Genetic.terminate
                return [MagicMock(value=10)]
        return MockParams()

def test_routing_problem_shortest_route_evaluation():
    constraint_list = basic_test_func.generate_dummy_constraint_list()
    problem = RoutingProblem(
        departure_time=None,
        boat=MockBoat(),
        constraint_list=constraint_list,
        fitness_function_type='shortest_route'
    )
    
    # Create a dummy route
    # 2 points: start and end. 
    route = np.array([
        [0, 0],
        [0, 1]
    ])
    # x is (1, n_points, 2)
    x = np.array([route])
    
    out = {}
    problem._evaluate(x, out)
    
    # Distance from (0,0) to (0,1) is roughly 111000m
    # The output should be distance
    
    print(f"DEBUG: out['F'] = {out['F']}")
    
    # Verify it's capturing distance logic
    # We can check if it's NOT fuel.
    # Fuel = distance/speed * rate = 111000/10 * 100 = 1110000
    # Distance = 111000
    # Values are distinct enough (factor of 10)
    
    val = out['F'][0][0]
    if hasattr(val, 'value'):
        val = val.value
    assert np.isclose(val, 111000, rtol=0.1)
    
    assert problem.fitness_function_type == 'shortest_route'
    
    # Re-instantiate with 'fuel'
    problem_fuel = RoutingProblem(
        departure_time=None,
        boat=MockBoat(),
        constraint_list=constraint_list,
        fitness_function_type='fuel'
    )
    assert problem_fuel.fitness_function_type == 'fuel'
    
    out_fuel = {}
    problem_fuel._evaluate(x, out_fuel)
    val_fuel = out_fuel['F'][0][0]
    if hasattr(val_fuel, 'value'):
        val_fuel = val_fuel.value
    assert np.isclose(val_fuel, 1110000, rtol=0.1)

def test_genetic_factory_initialization():
    class MockConfig:
        ALGORITHM_TYPE = 'genetic_shortest_route'
        GENETIC_NUMBER_GENERATIONS = 1
        GENETIC_NUMBER_OFFSPRINGS = 1
        GENETIC_POPULATION_SIZE = 1
        GENETIC_POPULATION_TYPE = 'grid_based'
        DEFAULT_MAP = [0, 0, 10, 10]
        GENETIC_FIX_RANDOM_SEED = False
        BOAT_TYPE = 'speedy_isobased' 
        DEFAULT_ROUTE = [0,0, 10,10]
        DEPARTURE_TIME = '2023-01-01'

    config = MockConfig()
    # verify Genetic init doesn't crash
    genetic_alg = Genetic(config)
    pass
