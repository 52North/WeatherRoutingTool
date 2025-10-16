"""CLI entry point for Weather Routing Tool."""
import argparse
import sys
import warnings
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    pass  # python-dotenv not installed, skip .env loading

from WeatherRoutingTool.execute_routing import execute_routing
from WeatherRoutingTool.config import Config, set_up_logging
from WeatherRoutingTool._version import __version__


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Weather Routing Tool - Optimize ship routes based on weather conditions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  wrt -f config.json
  wrt --validate-config config.json
  wrt -f config.json --debug --dry-run
        """)
    
    parser.add_argument('--version', action='version', version=f'WeatherRoutingTool {__version__}')
    parser.add_argument('-f', '--file', help="Config file path (absolute or relative)", type=str)
    parser.add_argument('--validate-config', help="Validate config file and exit", action='store_true')
    parser.add_argument('--dry-run', help="Validate config and setup without executing routing", action='store_true')
    parser.add_argument('--warnings-log-file',
                        help="Log file path for warnings and above", type=str)
    parser.add_argument('--info-log-file',
                        help="Log file path for info and above", type=str)
    parser.add_argument('--debug', help="Enable debug mode", action='store_true')
    parser.add_argument('--filter-warnings', 
                        help="Warning filter action",
                        choices=['default', 'error', 'ignore', 'always', 'module', 'once'],
                        default='default')
    
    args = parser.parse_args()
    
    # Set warning filter early
    warnings.filterwarnings(args.filter_warnings)
    
    # Validate-only mode
    if args.validate_config:
        if not args.file:
            parser.error("--validate-config requires -f/--file argument")
        try:
            config = Config.assign_config(Path(args.file))
            print(f"✓ Config validation successful: {args.file}")
            print(f"  Algorithm: {config.ALGORITHM_TYPE}")
            print(f"  Boat type: {config.BOAT_TYPE}")
            print(f"  Route: {config.DEFAULT_ROUTE}")
            return 0
        except Exception as e:
            print(f"✗ Config validation failed: {e}", file=sys.stderr)
            return 1
    
    # Normal execution requires config file
    if not args.file:
        parser.error("-f/--file is required (unless using --validate-config)")
    
    # Initialize logging
    set_up_logging(args.info_log_file, args.warnings_log_file, args.debug)
    
    # Load and validate config
    try:
        config = Config.assign_config(Path(args.file))
    except Exception as e:
        print(f"Failed to load config: {e}", file=sys.stderr)
        return 1
    
    # Dry-run mode: setup validation only
    if args.dry_run:
        print("Dry-run mode: Config loaded successfully, skipping execution.")
        return 0
    
    # Execute routing
    try:
        execute_routing(config)
        return 0
    except Exception as e:
        print(f"Routing execution failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
