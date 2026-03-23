import numpy as np
from pathlib import Path
from astropy import units as u
from WeatherRoutingTool.config import Config
from WeatherRoutingTool.ship.ship_config import ShipConfig
from WeatherRoutingTool.ship.ship_factory import ShipFactory
from WeatherRoutingTool.weather_factory import WeatherFactory
from WeatherRoutingTool.constraints.constraints import ConstraintsListFactory
from WeatherRoutingTool.utils.maps import Map
from WeatherRoutingTool.algorithms.genetic.problem import RoutingProblem

class TestRoutingProblem:
    def test_get_power_returns_positive_fuel(self):
        config_path = str(Path(__file__).parent.parent / 'config.json')
        config = Config.assign_config(config_path)
        ship_config = ShipConfig.assign_config(config_path)

        lat1, lon1, lat2, lon2 = config.DEFAULT_MAP
        default_map = Map(lat1, lon1, lat2, lon2)

        wt = WeatherFactory.get_weather(
            config._DATA_MODE_WEATHER,
            config.WEATHER_DATA,
            config.DEPARTURE_TIME,
            config.TIME_FORECAST,
            config.DELTA_TIME_FORECAST,
            default_map
        )

        boat = ShipFactory.get_ship(config.BOAT_TYPE, ship_config)

        constraint_list = ConstraintsListFactory.get_constraints_list(
            constraints_string_list=config.CONSTRAINTS_LIST,
            data_mode=config._DATA_MODE_DEPTH,
            min_depth=boat.get_required_water_depth(),
            map_size=default_map,
            depthfile=config.DEPTH_DATA,
            waypoints=config.INTERMEDIATE_WAYPOINTS,
            courses_path=config.COURSES_FILE
        )

        boat_speed = config.BOAT_SPEED * u.meter / u.second

        problem = RoutingProblem(
            departure_time=config.DEPARTURE_TIME,
            arrival_time=config.ARRIVAL_TIME,
            boat=boat,
            boat_speed=boat_speed,
            constraint_list=constraint_list
        )

        test_route = np.array([
            [10.0, 20.0, 3.09],
            [10.2, 20.2, 3.09],
            [10.4, 20.4, 3.09],
            [10.7, 20.7, 3.09]
        ])

        fuel, ship_params = problem.get_power(test_route)
        speed = ship_params.get_speed()

        assert fuel > 0, f'Expected positive fuel, got {fuel}'
        assert np.all(speed > 0), f'Expected all speeds positive, got {speed}'
        print(f'PASS - fuel={fuel:.2f} kg, speed mean={np.mean(speed):.4f} m/s')

if __name__ == '__main__':
    t = TestRoutingProblem()
    t.test_get_power_returns_positive_fuel()
    print('All tests passed!')
