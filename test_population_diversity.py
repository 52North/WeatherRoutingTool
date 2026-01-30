#!/usr/bin/env python3
"""
Test script to verify the population diversity improvement in genetic algorithms.
This script tests that the perturbation mechanism creates diverse routes instead of identical clones.
"""

import sys
import numpy as np
from unittest.mock import Mock, patch

# Add the WeatherRoutingTool to the path
sys.path.insert(0, '/Users/riteshthakur/Developer/open source/WeatherRoutingTool')

def test_isofuel_population_diversity():
    """Test that IsoFuelPopulation creates diverse routes when fallback is used."""
    print("Testing IsoFuelPopulation diversity...")
    
    try:
        from WeatherRoutingTool.algorithms.genetic.population import IsoFuelPopulation
        from WeatherRoutingTool.config import Config
        from WeatherRoutingTool.constraints.constraints import ConstraintsList
        
        # Create mock config
        mock_config = Mock(spec=Config)
        mock_config.DEPARTURE_TIME = None
        mock_config.ARRIVAL_TIME = None
        mock_config.BOAT_SPEED = 10.0
        mock_config.DEFAULT_ROUTE = [0.0, 0.0, 1.0, 1.0, 1.0]
        
        # Create mock constraints
        mock_constraints = Mock(spec=ConstraintsList)
        
        # Create population instance
        pop = IsoFuelPopulation(
            config=mock_config,
            default_route=[0.0, 0.0, 1.0, 1.0, 1.0],
            constraints_list=mock_constraints,
            pop_size=10
        )
        
        # Mock the patcher to return only 2 routes instead of 10
        mock_route1 = np.array([
            [0.0, 0.0, 10.0],  # start
            [0.5, 0.5, 10.0],  # intermediate
            [1.0, 1.0, 10.0]   # end
        ])
        
        mock_route2 = np.array([
            [0.0, 0.0, 10.0],  # start
            [0.6, 0.4, 10.0],  # intermediate
            [1.0, 1.0, 10.0]   # end
        ])
        
        pop.patcher = Mock()
        pop.patcher.patch.return_value = [mock_route1, mock_route2]
        
        # Generate population
        problem = Mock()
        X = pop.generate(problem, n_samples=10)
        
        # Verify we have 10 routes
        assert X.shape[0] == 10, f"Expected 10 routes, got {X.shape[0]}"
        
        # Check diversity: routes should not be identical
        unique_routes = []
        for i in range(10):
            route = X[i, 0]
            # Convert to tuple for comparison
            route_tuple = tuple(map(tuple, route))
            unique_routes.append(route_tuple)
        
        # Count unique routes
        unique_count = len(set(unique_routes))
        print(f"Generated {unique_count} unique routes out of 10")
        
        # With perturbation, we should have more than 2 unique routes
        assert unique_count > 2, f"Expected more than 2 unique routes, got {unique_count}"
        
        # Verify start and end points are preserved
        for i in range(10):
            route = X[i, 0]
            np.testing.assert_array_equal(route[0], [0.0, 0.0, 10.0], 
                                          err_msg=f"Route {i} start point modified")
            np.testing.assert_array_equal(route[-1], [1.0, 1.0, 10.0], 
                                          err_msg=f"Route {i} end point modified")
        
        print("✓ IsoFuelPopulation diversity test passed!")
        return True
        
    except Exception as e:
        print(f"✗ IsoFuelPopulation diversity test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_perturbation_mechanism():
    """Test the _perturb_route method directly."""
    print("\nTesting perturbation mechanism...")
    
    try:
        from WeatherRoutingTool.algorithms.genetic.population import IsoFuelPopulation
        from WeatherRoutingTool.config import Config
        from WeatherRoutingTool.constraints.constraints import ConstraintsList
        
        # Create population instance
        mock_config = Mock(spec=Config)
        mock_constraints = Mock(spec=ConstraintsList)
        
        pop = IsoFuelPopulation(
            config=mock_config,
            default_route=[0.0, 0.0, 1.0, 1.0, 1.0],
            constraints_list=mock_constraints,
            pop_size=10
        )
        
        # Create test route
        original_route = np.array([
            [0.0, 0.0, 10.0],  # start
            [0.5, 0.5, 10.0],  # intermediate 1
            [0.7, 0.7, 10.0],  # intermediate 2
            [1.0, 1.0, 10.0]   # end
        ])
        
        # Apply perturbations with different factors
        perturbed1 = pop._perturb_route(np.copy(original_route), 1)
        perturbed2 = pop._perturb_route(np.copy(original_route), 2)
        perturbed3 = pop._perturb_route(np.copy(original_route), 3)
        
        # Check that routes are different
        assert not np.array_equal(perturbed1, original_route), "Perturbed route 1 should be different"
        assert not np.array_equal(perturbed2, original_route), "Perturbed route 2 should be different"
        assert not np.array_equal(perturbed3, original_route), "Perturbed route 3 should be different"
        
        # Check that perturbed routes are different from each other
        assert not np.array_equal(perturbed1, perturbed2), "Perturbed routes should be different"
        assert not np.array_equal(perturbed2, perturbed3), "Perturbed routes should be different"
        
        # Verify start and end points are preserved
        np.testing.assert_array_equal(perturbed1[0], original_route[0], "Start point should be preserved")
        np.testing.assert_array_equal(perturbed1[-1], original_route[-1], "End point should be preserved")
        
        # Verify coordinates are within valid bounds
        assert np.all(perturbed1[:, 0] >= -90) and np.all(perturbed1[:, 0] <= 90), "Latitude out of bounds"
        assert np.all(perturbed1[:, 1] >= -180) and np.all(perturbed1[:, 1] <= 180), "Longitude out of bounds"
        
        print("✓ Perturbation mechanism test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Perturbation mechanism test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test edge cases for perturbation."""
    print("\nTesting edge cases...")
    
    try:
        from WeatherRoutingTool.algorithms.genetic.population import IsoFuelPopulation
        from WeatherRoutingTool.config import Config
        from WeatherRoutingTool.constraints.constraints import ConstraintsList
        
        # Create population instance
        mock_config = Mock(spec=Config)
        mock_constraints = Mock(spec=ConstraintsList)
        
        pop = IsoFuelPopulation(
            config=mock_config,
            default_route=[0.0, 0.0, 1.0, 1.0, 1.0],
            constraints_list=mock_constraints,
            pop_size=10
        )
        
        # Test with route that has only start and end points
        short_route = np.array([
            [0.0, 0.0, 10.0],  # start
            [1.0, 1.0, 10.0]   # end
        ])
        
        perturbed_short = pop._perturb_route(np.copy(short_route), 1)
        
        # Should return unchanged since there are no intermediate waypoints
        np.testing.assert_array_equal(perturbed_short, short_route, 
                                      "Short route should remain unchanged")
        
        print("✓ Edge cases test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Edge cases test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Population Diversity Improvement Test")
    print("=" * 50)
    
    tests = [
        test_isofuel_population_diversity,
        test_perturbation_mechanism,
        test_edge_cases,
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
        print("🎉 All tests passed! Population diversity improvement works correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
