import argparse
import logging
import sys
import warnings

from WeatherRoutingTool.config import Config, set_up_logging
from WeatherRoutingTool.execute_routing import execute_routing
from WeatherRoutingTool.ship.ship_config import ShipConfig
from WeatherRoutingTool.utils.errors import WRTErrorCodes, WRTError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Weather Routing Tool')
    parser.add_argument('-f', '--file', help="Config file name (absolute path)", required=True, type=str)
    parser.add_argument('--warnings-log-file',
                        help="Logging file name (absolute path) for warnings and above.", required=False, type=str)
    parser.add_argument('--info-log-file',
                        help="Logging file name (absolute path) for info and above.", required=False, type=str)
    parser.add_argument('--debug', help="Enable debug mode. <True|False>. Defaults to 'False'.",
                        required=False, type=str, default='False')
    parser.add_argument('--filter-warnings', help="Filter action. <default|error|ignore|always|module|once>."
                        "Defaults to 'default'.", required=False, type=str, default='default')
    args = parser.parse_args()
    if not args.file:
        raise RuntimeError("No config file name provided!")
    debug_str = str(args.debug).lower()
    if debug_str == 'true':
        args.debug = True
    elif debug_str == 'false':
        args.debug = False
    else:
        raise ValueError("--debug does not have a valid value")
    if args.filter_warnings not in ['default', 'error', 'ignore', 'always', 'module', 'once']:
        raise ValueError("--filter-warnings has to be one of <default|error|ignore|always|module|once>")

    ##
    # initialise logging
    set_up_logging(args.info_log_file, args.warnings_log_file, args.debug)

    logger = logging.getLogger('WRT')
    try:
        config = Config.assign_config(args.file)
        ship_config = ShipConfig.assign_config(args.file)
        execute_routing(config, ship_config)
        error_code = WRTErrorCodes.NO_ERROR
    except WRTError as err:
        logger.error(str(err))
        logger.debug('Traceback:', exc_info=True)
        error_code = err.code
    except Exception:
        logger.exception('Unexpected error of the Weather Routing Tool.')
        error_code = WRTErrorCodes.INTERNAL

    # set warning filter action (https://docs.python.org/3/library/warnings.html)
    warnings.filterwarnings(args.filter_warnings)

    sys.exit(int(error_code))
