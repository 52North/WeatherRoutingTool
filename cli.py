import argparse
import warnings

from WeatherRoutingTool.execute_routing import execute_routing
from WeatherRoutingTool.config import Config, set_up_logging

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

    ##
    # create config object
    config_obj = Config(file_name=args.file)
    config_obj.print()

    ##
    # set warning filter action (https://docs.python.org/3/library/warnings.html)
    warnings.filterwarnings(args.filter_warnings)

    ##
    # run route optimization
    execute_routing(config_obj)
