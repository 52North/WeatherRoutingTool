from WeatherRoutingTool.environmental_data.weather_factory import WeatherFactory

class Mediator():
    """
    Experimental Mediator class
    """
    _weather: WeatherCond

    def __init__(self, config):
        config = Config(file_name=args.file)
        windfile = config.WEATHER_DATA
        time_resolution = config.DELTA_TIME_FORECAST
        time_forecast = config.TIME_FORECAST
        lat1, lon1, lat2, lon2 = config.DEFAULT_MAP
        departure_time = datetime.strptime(config.DEPARTURE_TIME, '%Y-%m-%dT%H:%MZ')
        default_map = Map(lat1, lon1, lat2, lon2)

        self._weather = WeatherFactory.get_weather(config.DATA_MODE, windfile, departure_time, time_forecast, time_resolution,
                                    default_map)
        pass

    def init_modules(self):
        pass

    def get_weather(self, parameter_list):
        self._weather.get_weather()

