"""Command-line interface for Weather Routing Tool."""

import argparse
import logging
import sys
from pathlib import Path

from WeatherRoutingTool.config import Config, set_up_logging
from WeatherRoutingTool.algorithms.routingalg_factory import RoutingAlgFactory
from WeatherRoutingTool.constraints.constraints import ConstraintsListFactory
from WeatherRoutingTool.ship.ship_factory import ShipFactory
from WeatherRoutingTool.utils.maps import Map
from WeatherRoutingTool.weather_factory import WeatherFactory

logger = logging.getLogger('WRT.CLI')


def main():
    """Main entry point for the WRT CLI."""
    parser = argparse.ArgumentParser(
        description='Weather Routing Tool - Optimize ship routes based on weather conditions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  wrt --config config.json
  wrt --config config.json --verbose
  wrt --config config.json --log-file routing.log
        """
    )
    
    parser.add_argument(
        '--config',
        required=True,
        type=Path,
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose (DEBUG) logging'
    )
    parser.add_argument(
        '--log-file',
        type=Path,
        help='Write logs to file (in addition to stdout)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    set_up_logging(
        info_log_file=str(args.log_file) if args.log_file else None,
        debug=args.verbose,
        log_level=log_level
    )
    
    logger.info("=" * 80)
    logger.info("Weather Routing Tool (WRT)")
    logger.info("=" * 80)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = Config.assign_config(path=args.config, init_mode='from_json')
        logger.info("Configuration loaded successfully")
        
        # Initialize components
        logger.info("Initializing routing components...")
        
        # Map
        lat1, lon1, lat2, lon2 = config.DEFAULT_MAP
        default_map = Map(lat1, lon1, lat2, lon2)
        logger.info(f"Map bounds: ({lat1}, {lon1}) to ({lat2}, {lon2})")
        
        # Weather
        weather = WeatherFactory.get_weather(
            data_mode=config._DATA_MODE_WEATHER,
            weatherfile=config.WEATHER_DATA,
            departure_time=config.DEPARTURE_TIME,
            time_forecast=config.TIME_FORECAST,
            delta_time_forecast=config.DELTA_TIME_FORECAST,
            default_map=default_map
        )
        logger.info("Weather data loaded")
        
        # Constraints
        constraint_kwargs = {
            'map_size': default_map,
        }
        if 'water_depth' in config.CONSTRAINTS_LIST:
            constraint_kwargs.update({
                'data_mode': config._DATA_MODE_DEPTH,
                'min_depth': 20,  # TODO: make configurable
                'depthfile': config.DEPTH_DATA
            })
        if 'via_waypoints' in config.CONSTRAINTS_LIST:
            constraint_kwargs['waypoints'] = config.INTERMEDIATE_WAYPOINTS
            
        constraints_list = ConstraintsListFactory.get_constraints_list(
            config.CONSTRAINTS_LIST,
            **constraint_kwargs
        )
        logger.info(f"Constraints configured: {', '.join(config.CONSTRAINTS_LIST)}")
        
        # Ship
        boat = ShipFactory.get_boat(config.BOAT_TYPE, config)
        logger.info(f"Boat type: {config.BOAT_TYPE}")
        
        # Routing algorithm
        routing_alg = RoutingAlgFactory.get_algorithm(config.ALGORITHM_TYPE, config)
        logger.info(f"Algorithm: {config.ALGORITHM_TYPE}")
        
        # Execute routing
        logger.info("=" * 80)
        logger.info("Starting route optimization...")
        logger.info("=" * 80)
        
        routing_alg.execute_routing(boat, weather, constraints_list, verbose=args.verbose)
        
        # Terminate and save
        routing_alg.terminate()
        
        logger.info("=" * 80)
        logger.info("Route optimization completed successfully!")
        logger.info(f"Results saved to: {config.ROUTE_PATH}")
        logger.info("=" * 80)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error during routing: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
