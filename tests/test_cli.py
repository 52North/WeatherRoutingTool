import unittest
from unittest.mock import patch, MagicMock, call
import pytest
from cli import main


class TestCLI(unittest.TestCase):

    @patch('cli.execute_routing')
    @patch('cli.ShipConfig.assign_config')
    @patch('cli.Config.assign_config')
    @patch('cli.set_up_logging')
    @patch('cli.warnings.filterwarnings')
    def test_cli_main_valid_args(self, mock_filterwarnings, mock_logging, mock_config, mock_ship_config, mock_execute):
        mock_config.return_value = MagicMock()
        mock_ship_config.return_value = MagicMock()

        args = ['-f', 'config.json', '--debug', 'true', '--filter-warnings', 'ignore']
        main(args)

        mock_filterwarnings.assert_called_once_with('ignore')
        mock_logging.assert_called_once_with(None, None, True)
        mock_config.assert_called_once_with('config.json')
        mock_ship_config.assert_called_once_with('config.json')
        mock_execute.assert_called_once_with(mock_config.return_value, mock_ship_config.return_value)

    @patch('cli.execute_routing')
    @patch('cli.ShipConfig.assign_config')
    @patch('cli.Config.assign_config')
    @patch('cli.set_up_logging')
    @patch('cli.warnings.filterwarnings')
    def test_cli_warning_filter_applied_before_routing(
            self, mock_filterwarnings, mock_logging, mock_config, mock_ship_config, mock_execute):
        manager = MagicMock()
        manager.attach_mock(mock_filterwarnings, 'filterwarnings')
        manager.attach_mock(mock_execute, 'execute_routing')

        args = ['-f', 'config.json', '--filter-warnings', 'always']
        main(args)

        expected_calls = [
            call.filterwarnings('always'),
            call.execute_routing(mock_config.return_value, mock_ship_config.return_value)
        ]
        self.assertEqual(manager.mock_calls, expected_calls)

    def test_cli_invalid_debug_raises_value_error(self):
        args = ['-f', 'config.json', '--debug', 'invalid_bool']
        with pytest.raises(ValueError, match="--debug does not have a valid value"):
            main(args)

    def test_cli_invalid_filter_warnings_raises_value_error(self):
        args = ['-f', 'config.json', '--filter-warnings', 'invalid_action']
        with pytest.raises(ValueError, match="--filter-warnings has to be one of"):
            main(args)


if __name__ == '__main__':
    unittest.main()
