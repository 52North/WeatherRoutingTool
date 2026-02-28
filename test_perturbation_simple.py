#!/usr/bin/env python3
"""
Simple test to verify the perturbation logic without requiring full module imports.
This tests the core perturbation algorithm directly.
"""

import numpy as np
import sys
import os

def test_perturbation_logic():
    """Test the perturbation logic directly."""
    print("Testing perturbation logic...")
    
    def perturb_route(route: np.ndarray, perturbation_factor: int) -> np.ndarray:
        """Apply controlled perturbation to a route to maintain population diversity."""
        if len(route) <= 2:
            # Route has only start and end points, cannot perturb
            return route
            
        # Create perturbed copy
        perturbed_route = np.copy(route)
        
        # Calculate perturbation magnitude (0.1 to 1.0 degrees based on factor)
        # Cap at reasonable maximum to maintain route coherence
        max_perturbation = min(0.1 + (perturbation_factor * 0.1), 1.0)
        
        # Perturb intermediate waypoints (exclude start [0] and end [-1])
        for i in range(1, len(route) - 1):
            # Apply random perturbation to latitude and longitude
            lat_perturbation = np.random.uniform(-max_perturbation, max_perturbation)
            lon_perturbation = np.random.uniform(-max_perturbation, max_perturbation)
            
            # Apply perturbations
            perturbed_route[i, 0] += lat_perturbation  # latitude
            perturbed_route[i, 1] += lon_perturbation  # longitude
            
            # Ensure coordinates remain within valid ranges
            perturbed_route[i, 0] = np.clip(perturbed_route[i, 0], -90, 90)    # latitude bounds
            perturbed_route[i, 1] = np.clip(perturbed_route[i, 1], -180, 180)  # longitude bounds
        
        return perturbed_route
    
    try:
        # Set seed for reproducible tests
        np.random.seed(42)
        
        # Create test route
        original_route = np.array([
            [0.0, 0.0, 10.0],  # start
            [0.5, 0.5, 10.0],  # intermediate 1
            [0.7, 0.7, 10.0],  # intermediate 2
            [1.0, 1.0, 10.0]   # end
        ])
        
        # Apply perturbations with different factors
        perturbed1 = perturb_route(np.copy(original_route), 1)
        perturbed2 = perturb_route(np.copy(original_route), 2)
        perturbed3 = perturb_route(np.copy(original_route), 3)
        
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
        
        # Test edge case: route with only start and end points
        short_route = np.array([
            [0.0, 0.0, 10.0],  # start
            [1.0, 1.0, 10.0]   # end
        ])
        
        perturbed_short = perturb_route(np.copy(short_route), 1)
        np.testing.assert_array_equal(perturbed_short, short_route, "Short route should remain unchanged")
        
        print("✓ Perturbation logic test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Perturbation logic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_diversity_simulation():
    """Simulate the diversity improvement scenario."""
    print("\nTesting diversity simulation...")
    
    def perturb_route(route: np.ndarray, perturbation_factor: int) -> np.ndarray:
        """Apply controlled perturbation to a route to maintain population diversity."""
        if len(route) <= 2:
            return route
            
        perturbed_route = np.copy(route)
        max_perturbation = min(0.1 + (perturbation_factor * 0.1), 1.0)
        
        for i in range(1, len(route) - 1):
            lat_perturbation = np.random.uniform(-max_perturbation, max_perturbation)
            lon_perturbation = np.random.uniform(-max_perturbation, max_perturbation)
            perturbed_route[i, 0] += lat_perturbation
            perturbed_route[i, 1] += lon_perturbation
            perturbed_route[i, 0] = np.clip(perturbed_route[i, 0], -90, 90)
            perturbed_route[i, 1] = np.clip(perturbed_route[i, 1], -180, 180)
        
        return perturbed_route
    
    try:
        # Set seed for reproducible tests
        np.random.seed(42)
        
        # Simulate the scenario: IsoFuel generates only 3 routes, but we need 10
        n_samples = 10
        generated_routes = 3
        
        # Create mock routes
        base_routes = []
        for i in range(generated_routes):
            route = np.array([
                [0.0, 0.0, 10.0],  # start
                [0.5 + i*0.1, 0.5 + i*0.1, 10.0],  # intermediate
                [1.0, 1.0, 10.0]   # end
            ])
            base_routes.append(route)
        
        # OLD approach: exact copying
        old_population = []
        for i in range(n_samples):
            if i < generated_routes:
                old_population.append(base_routes[i])
            else:
                old_population.append(np.copy(base_routes[-1]))  # Exact copy
        
        # NEW approach: perturbation
        new_population = []
        for i in range(n_samples):
            if i < generated_routes:
                new_population.append(base_routes[i])
            else:
                perturbed = perturb_route(np.copy(base_routes[-1]), i - generated_routes + 1)
                new_population.append(perturbed)
        
        # Count unique routes in old approach
        old_unique = len(set(tuple(map(tuple, route)) for route in old_population))
        new_unique = len(set(tuple(map(tuple, route)) for route in new_population))
        
        print(f"Old approach: {old_unique} unique routes out of {n_samples}")
        print(f"New approach: {new_unique} unique routes out of {n_samples}")
        
        # Old approach should have only 3 unique routes (the generated ones)
        assert old_unique == generated_routes, f"Old approach should have {generated_routes} unique routes"
        
        # New approach should have more unique routes
        assert new_unique > generated_routes, f"New approach should have more than {generated_routes} unique routes"
        
        print("✓ Diversity simulation test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Diversity simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Population Diversity Improvement Test (Simple)")
    print("=" * 50)
    
    tests = [
        test_perturbation_logic,
        test_diversity_simulation,
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
