import copy
import logging
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from astropy import units as u

import WeatherRoutingTool.utils.formatting as form
import WeatherRoutingTool.utils.unit_conversion as units
from WeatherRoutingTool.ship.ship import Boat
from WeatherRoutingTool.ship.shipparams import ShipParams
from WeatherRoutingTool.ship.ship_config import ShipConfig

have_maripower = False
try:
    import mariPower
    from mariPower import __main__

    have_maripower = True
except ModuleNotFoundError:
    pass  # maripower installation is optional

logger = logging.getLogger('WRT.ship')


##
# Class implementing connection to mariPower package.
#
# 'Flow' of information:
# 1) Before starting the routing procedure, the routing tool writes the environmental data to a netCDF file (in the
# following: 'EnvData netCDF').
# 2) The routing tool (WRT) writes the courses per space-time point to a netCDF file (in the following: 'courses
# netCDF') for which the power consumption will be requested.
#       -> Tanker.write_netCDF_courses
# 3) The WRT sends the paths to the 'EnvData netCDF' and the 'courses netCDF' to mariPower and requests the power
# calculation.
#       -> Tanker.get_fuel_netCDF_loop
# 4) The mariPower package writes the results for the power estimation to the 'courses netCDF'.
# 5) The WRT extracts the power from the 'courses netCDF'.
#       -> Tanker.extract_fuel_from_netCDF
#
# Steps 1), 3), and 5) are combined in the function
#       -> Tanker.get_fuel_per_time_netCDF
#
#
# Functions that are named something like *simple_fuel* are meant to be used as placeholders for the mariPower
# package. They should only be used for
# testing purposes.

class MariPowerTanker(Boat):
    # Connection to hydrodynamic modeling
    # hydro_model: mariPower.ship
    draught: float

    # additional information
    courses_path: str  # path to netCDF which contains the power estimation per course
    depth_path: str  # path to netCDF for depth data
    # FIXME: make separate weather path obsolete
    weather_path_maripower: str  # path to weather data which is converted to maripower requirements
    courses_path_maripower: str

    use_depth_data: bool
    correct_by_nnmodel: bool

    def __init__(self, ship_config: ShipConfig):
        super().__init__(ship_config)

        if not have_maripower:
            raise ModuleNotFoundError(
                'You are trying to use maripower to estimate the fuel consumption but maripower is'
                ' not installed. Please install the necessary software.')

        # mandatory variables for maripower
        if not ship_config.COURSES_FILE:
            raise Exception('COURSES_FILE is a mandatory parameter for the maripower tanker!')

        self.courses_path = str(ship_config.COURSES_FILE)
        self.weather_path = str(ship_config.WEATHER_DATA)
        self.courses_path_maripower = self.courses_path.removesuffix(".nc") + "_output.nc"

        # optional variables for maripower
        if ship_config.DEPTH_DATA is not None:
            self.use_depth_data = True
            self.depth_path = str(ship_config.DEPTH_DATA)

        self.hydro_model = mariPower.ship.CBT()
        self.hydro_model.Draught_AP = np.array([ship_config.BOAT_DRAUGHT_AFT])
        self.hydro_model.Draught_FP = np.array([ship_config.BOAT_DRAUGHT_FORE])
        self.hydro_model.Roughness_Level = np.array([ship_config.BOAT_ROUGHNESS_LEVEL])
        self.hydro_model.Roughness_Distribution_Level = np.array([ship_config.BOAT_ROUGHNESS_DISTRIBUTION_LEVEL])
        self.hydro_model.WindForcesFactor = ship_config.BOAT_FACTOR_WIND_FORCES
        self.hydro_model.WaveForcesFactor = ship_config.BOAT_FACTOR_WAVE_FORCES
        self.hydro_model.CalmWaterFactor = ship_config.BOAT_FACTOR_CALM_WATER

        # Fine-tuning the following parameters might lead to a significant speedup of mariPower. However, they should
        # be set carefully because the accuracy of the predictions might also be affected
        # self.hydro_model.MaxIterations = 25  # mariPower default: 10
        # self.hydro_model.Tolerance = 0.000001  # mariPower default: 0.0
        # self.hydro_model.Relaxation = 0.7  # mariPower default: 0.3

    def load_data(self):
        if self.use_depth_data:
            self.depth_data = mariPower.environment.EnvironmentalData_Depth(self.depth_path)

        self.weather_adapter()

    # FIXME: make weather adapter obsolete
    def weather_adapter(self):
        debug = False

        weather_str = str(self.weather_path).split('.nc')
        self.weather_path_maripower = weather_str[0] + '_maripower.nc'

        if os.path.isfile(self.weather_path_maripower):
            return

        if debug:
            print('maripower weather adapter: reading weather data from ', self.weather_path)
            print('writing converted data to ', self.weather_path_maripower)

        ds = xr.open_dataset(self.weather_path)

        thetao = ds['thetao']
        so = ds['so']
        ucurrent = ds['utotal']
        vcurrent = ds['vtotal']

        ds_cut = ds.drop_vars(['thetao', 'so', 'utotal', 'vtotal'])
        ds_cut = ds_cut.drop_dims('depth')

        thetao = thetao.isel(depth=0)
        so = so.isel(depth=0)
        ucurrent = ucurrent.isel(depth=0)
        vcurrent = vcurrent.isel(depth=0)

        ds_cut = xr.merge([ds_cut, thetao])
        ds_cut = xr.merge([ds_cut, so])
        ds_cut = xr.merge([ds_cut, ucurrent])
        ds_cut = xr.merge([ds_cut, vcurrent])

        if debug:
            print('new data: ', ds_cut)

        ds_cut.to_netcdf(self.weather_path_maripower)
        ds_cut.close()
        ds.close()

    def set_ship_property(self, variable, value):
        print('Setting ship property ' + variable + ' to ' + str(value))
        setattr(self.hydro_model, variable, value)

    def print_init(self):
        logger.info(form.get_log_step('path to weather data' + str(self.weather_path), 1))
        logger.info(form.get_log_step('path to courses data' + str(self.courses_path), 1))
        logger.info(form.get_log_step('path to depth data' + str(self.depth_path), 1))

    def set_env_data_path(self, path):
        self.weather_path_maripower = path

    def set_courses_path(self, path):
        self.courses_path = path

    ##
    # Writes netCDF which stores courses in dependence on latitude, longitude and time for further processing by
    # mariPower.
    # Several courses can be provided per space point. In this case, the arrays lats and lons need to be filled
    # e.g. power estimation is requested for 3 courses (c1, c2, c3) for 1 space-time point (lat1, lon1) then:
    #   courses = {c1, c2, c3}
    #   lats = {lat1, lat1, lat1}
    #   lons = {lon1, lon1, lon1}

    def write_netCDF_courses(self, courses, lats, lons, time, speed, unique_coords=False):
        debug = False

        # ToDo: use logger.debug and args.debug
        if (debug):
            print('Requesting power calculation')
            time_str = 'Time:' + str(time.shape)
            lats_str = 'Latitude:' + str(lats.shape)
            lons_str = 'Longitude:' + str(lons.shape)
            course_str = 'Courses:' + str(courses.shape)
            speed_str = 'Boat speed:' + str(speed.shape)
            form.print_step(time_str, 1)
            form.print_step(lats_str, 1)
            form.print_step(lons_str, 1)
            form.print_step(course_str, 1)
            form.print_step(speed_str, 1)
        n_coords = None
        if unique_coords:
            it = sorted(np.unique(lons, return_index=True)[1])
            lons = lons[it]
            lats = lats[it]
            n_coords = lons.shape[0]
        else:
            n_coords = lons.shape[0]
        # number or coordinate pairs
        n_courses = int(courses.shape[0] / n_coords)  # number of courses per coordinate pair

        # generate iterator for courses and coordinate pairs
        it_course = np.arange(n_courses) + 1  # get iterator with a length of the number of unique longitude values
        it_course = np.hstack(
            (it_course,) * n_coords)  # prepare iterator for Data Frame: repeat each element as often as
        it_pos = np.arange(n_coords) + 1
        it_pos = np.repeat(it_pos, n_courses)  # np.hstack((it_pos,) * n_courses)

        if (debug):
            form.print_step('it_course=' + str(it_course))
            form.print_step('it_pos=' + str(it_pos))

        assert courses.shape == it_pos.shape
        assert courses.shape == it_course.shape
        assert courses.shape == speed.shape

        # generate pandas DataFrame
        df = pd.DataFrame({'it_pos': it_pos, 'it_course': it_course, 'courses': courses, 'speed': speed, })

        df = df.set_index(['it_pos', 'it_course'])
        # ToDo: use logger.debug and args.debug
        if (debug):
            print('pandas DataFrame:', df)

        ds = df.to_xarray()

        time_reshape = time.reshape(ds['it_pos'].shape[0], ds['it_course'].shape[0])[:, 0]

        logger.info('Request power calculation for ' + str(n_courses) + ' courses and ' + str(n_coords) +
                    ' coordinates')

        ds["lon"] = (['it_pos'], lons)
        ds["lat"] = (['it_pos'], lats)
        ds["time"] = (['it_pos'], time_reshape)
        assert ds['lon'].shape == ds['lat'].shape
        assert ds['time'].shape == ds['lat'].shape

        # ToDo: use logger.debug and args.debug
        if (debug):
            print('xarray DataSet', ds)

        ds.to_netcdf(self.courses_path)
        if (debug):
            ds_read = xr.open_dataset(self.courses_path)
            print('read data set', ds_read)
        ds.close()

    ##
    # extracts power from 'courses netCDF' which has been written by mariPower and returns it as 1D array.
    def extract_params_from_netCDF(self, ds):
        debug = False
        if (debug):
            form.print_step('Dataset with ship parameters:' + str(ds), 1)

        ds = xr.open_dataset(self.courses_path_maripower)

        power = ds['Power_brake'].to_numpy().flatten() * u.Watt
        rpm = ds['RotationRate'].to_numpy().flatten() * 1 / u.minute
        fuel = ds['Fuel_consumption_rate'].to_numpy().flatten() * u.tonne / u.hour
        fuel = fuel.to(u.kg / u.second)
        r_wind = ds['Wind_resistance'].to_numpy().flatten() * u.newton
        r_calm = ds['Calm_resistance'].to_numpy().flatten() * u.newton
        r_waves = ds['Wave_resistance'].to_numpy().flatten() * u.newton
        r_shallow = ds['Shallow_water_resistance'].to_numpy().flatten() * u.newton
        r_roughness = ds['Hull_roughness_resistance'].to_numpy().flatten() * u.newton
        status = ds['Status'].to_numpy().flatten()
        message = ds['Message'].to_numpy().flatten()
        speed = ds['speed'].to_numpy().flatten() * u.meter / u.second

        ship_params = ShipParams(
            fuel_rate=fuel,
            power=power,
            rpm=rpm,
            speed=speed,
            r_wind=r_wind,
            r_calm=r_calm,
            r_waves=r_waves,
            r_shallow=r_shallow,
            r_roughness=r_roughness,
            wave_height=-99,
            wave_direction=-99,
            wave_period=-99,
            u_currents=-99,
            v_currents=-99,
            u_wind_speed=-99,
            v_wind_speed=-99,
            pressure=-99,
            air_temperature=-99,
            salinity=-99,
            water_temperature=-99,
            status=status,
            message=message
        )

        if (debug):
            form.print_step('Dataset with fuel' + str(ds), 1)
            form.print_step('original shape power' + str(power.shape), 1)
            form.print_step('flattened shape power' + str(ship_params.get_power().shape), 1)
            form.print_step('power result' + str(ship_params.get_power()))

        ds.close()
        return ship_params

    ##
    # Passes paths for 'courses netCDF' and 'environmental data netCDF' to mariPower and request estimation of power
    # consumption.
    # Is not yet working as explained for Tanker.get_fuel_netCDF_loop
    #
    def get_fuel_netCDF(self):
        mariPower_ship = copy.deepcopy(self.hydro_model)
        if self.use_depth_data:
            status, message, envDataRoute = mariPower.helper_functions.PredictPowerOrSpeedRoute(
                mariPower_ship, self.courses_path, self.weather_path_maripower, self.depth_data)
        else:
            status, message, envDataRoute = mariPower.helper_functions.PredictPowerOrSpeedRoute(
                mariPower_ship, self.courses_path, self.weather_path_maripower)
        # form.print_current_time('time for mariPower request:', start_time)
        # ToDo: read messages from netCDF and store them in ship_params (changes in mariPower necessary)
        # for idx in range(0, len(status.flatten())):
        #     if status.flatten()[idx] != 1:
        #         logger.warning(f"{idx}: status.shape={status.shape}, status={status.flatten()[idx]}, "
        #                        f"message={message.flatten()[idx]}")

        ds_read = xr.open_dataset(self.courses_path)
        return ds_read

    ##
    # main function for communication with mariPower package (see documentation above)
    def get_ship_parameters(self, courses, lats, lons, time, speed, unique_coords=False):
        assert speed.shape == time.shape
        self.write_netCDF_courses(courses, lats, lons, time, speed, unique_coords)

        # ds = self.get_fuel_netCDF_loop()
        # ds = self.get_fuel_netCDF_dummy(ds, courses, wind)
        ds = self.get_fuel_netCDF()
        ship_params = self.extract_params_from_netCDF(ds)
        ship_params = self.evaluate_weather(ship_params, lats, lons, time)
        ds.close()

        return ship_params
