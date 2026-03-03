import logging

import WeatherRoutingTool.utils.formatting as form
from WeatherRoutingTool.weather import WeatherCondFromFile, WeatherCondEnvAutomatic, WeatherCondODC, FakeWeather

logger = logging.getLogger('WRT.weather')


class WeatherFactory:

    def __init__(self):
        pass

    @staticmethod
    def get_weather(data_mode, file_path, departure_time, time_forecast, time_resolution, default_map, **kwargs):
        wt = None

        form.print_line()
        logger.info('Initialising weather')

        if data_mode == 'skip':
            return None

        if data_mode == 'from_file':
            logger.info(form.get_log_step('Reading weather data from file:  ' + file_path, 0))
            wt = WeatherCondFromFile(departure_time, time_forecast, time_resolution)
            wt.set_map_size(default_map)
            wt.read_dataset(file_path)

        if data_mode == 'automatic':
            logger.info(form.get_log_step('Automatic download from weather data.', 0))
            wt_download = WeatherCondEnvAutomatic(departure_time, time_forecast, time_resolution)
            wt_download.set_map_size(default_map)
            wt_download.read_dataset()
            wt_download.write_data(file_path)

            wt = WeatherCondFromFile(departure_time, time_forecast, time_resolution)
            wt.read_dataset(file_path)

        if data_mode == 'odc':
            logger.info(form.get_log_step('Loading data with OpenDataCube.', 0))
            wt_download = WeatherCondODC(departure_time, time_forecast, time_resolution)
            wt_download.set_map_size(default_map)
            wt_download.read_dataset()
            wt_download.write_data(file_path)

            wt = WeatherCondFromFile(departure_time, time_forecast, time_resolution)
            wt.read_dataset(file_path)

        if data_mode == 'fake':
            var_dict = kwargs.get('var_dict')
            coord_res = kwargs.get('coord_res')
            gauß_dict = kwargs.get('gauß_dict')

            logger.info(form.get_log_step('Faking weather data.', 0))
            wt_download = FakeWeather(departure_time, time_forecast, time_resolution, coord_res, var_dict, gauß_dict)
            wt_download.set_map_size(default_map)
            wt_download.read_dataset()
            wt_download.write_data(file_path)

            wt = WeatherCondFromFile(departure_time, time_forecast, time_resolution)
            wt.set_map_size(default_map)
            wt.read_dataset(file_path)

        wt.check_units()
        wt.print_init()

        return wt
