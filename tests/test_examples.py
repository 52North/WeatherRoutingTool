from pathlib import Path
from unittest.mock import patch

from examples.genetic_examples import run_genetic_example


def test_genetic_example_importable():
    """Ensure run_genetic_example is callable and importable."""
    assert callable(run_genetic_example)


@patch("examples.genetic_examples.execute_routing")
def test_genetic_example_execution(mock_execute_routing):
    """Test that run_genetic_example correctly loads config and invokes execute_routing."""
    config_path = Path(__file__).parent / "config.tests.json"
    run_genetic_example(str(config_path))

    assert mock_execute_routing.called
    config_arg, ship_config_arg = mock_execute_routing.call_args[0]
    assert config_arg.ALGORITHM_TYPE == "genetic"
    assert ship_config_arg.BOAT_TYPE == "direct_power_method"
