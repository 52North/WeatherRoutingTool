"""
Tests for IsoFuelPopulation fallback behavior when patcher
returns fewer routes than requested.
"""



import numpy as np
import pytest

from algorithms.genetic.population import IsoFuelPopulation
class DummyPatcher:
    def generate(self, *args, **kwargs):
        # Return only ONE valid route
        route = np.array([[0.0, 0.0], [1.0, 1.0]])
        return [route]
def test_fallback_population_is_not_identical():
    n_samples = 5

    patcher = DummyPatcher()
    population = IsoFuelPopulation(
        n_samples=n_samples,
        patcher=patcher
    )

    X = population.generate()

    # Ensure correct population size
    assert len(X) == n_samples

    # Compare routes — they should NOT all be identical
    first = X[0, 0]
    for i in range(1, n_samples):
        assert not np.allclose(first, X[i, 0]), \
            "Fallback population contains identical clones"
