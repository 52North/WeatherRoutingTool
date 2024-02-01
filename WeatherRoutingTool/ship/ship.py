import copy
import logging
import math
import sys

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from astropy import units as u

import mariPower
import WeatherRoutingTool.utils.formatting as form
import WeatherRoutingTool.utils.unit_conversion as units
from mariPower import __main__
from WeatherRoutingTool.ship.shipparams import ShipParams

logger = logging.getLogger('WRT.ship')


# Boat: Main class for boats. Classes 'Tanker' and 'SailingBoat' derive from it
# Tanker: implements interface to mariPower package which is used for power estimation.

class Boat:
    speed: float  # boat speed in m/s

    def __init__(self, config):
        self.speed = config.BOAT_SPEED * u.meter/u.second
        pass

    def get_ship_parameters(self, courses, lats, lons, time, speed=None, unique_coords=False):
        pass

    def get_boat_speed(self):
        return self.speed

    def print_init(self):
        pass


class ConstantFuelBoat(Boat):
    fuel: float  # dummy value for fuel_rate that is returned
    speed: float    # boat speed

    def __init__(self, config):
        super().__init__(config)
        self.fuel = config.CONSTANT_FUEL_RATE * u.kg/u.second

    def print_init(self):
        logger.info(form.get_log_step('boat speed' + str(self.speed), 1))
        logger.info(form.get_log_step('boat fuel rate' + str(self.fuel), 1))

    def get_ship_parameters(self, courses, lats, lons, time, speed=None, unique_coords=False):
        debug = False
        n_requests = len(courses)

        dummy_array = np.full(n_requests, -99)
        fuel_array = np.full(n_requests, self.fuel)
        speed_array = np.full(n_requests, self.speed)

        ship_params = ShipParams(
            fuel=fuel_array,
            power=dummy_array,
            rpm=dummy_array,
            speed=speed_array,
            r_wind=dummy_array,
            r_calm=dummy_array,
            r_waves=dummy_array,
            r_shallow=dummy_array,
            r_roughness=dummy_array,
            status=dummy_array
        )

        if (debug):
            ship_params.print()
            form.print_step('fuel result' + str(ship_params.get_fuel_rate()))

        return ship_params


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

class Tanker(Boat):
    # Connection to hydrodynamic modeling
    hydro_model: mariPower.ship
    draught: float

    # additional information
    environment_path: str  # path to netCDF for environmental data
    courses_path: str  # path to netCDF which contains the power estimation per course
    depth_path: str  # path to netCDF for depth data

    use_depth_data: bool

    def __init__(self, config):
        super().__init__(config)
        self.environment_path = config.WEATHER_DATA
        self.courses_path = config.COURSES_FILE
        self.depth_path = config.DEPTH_DATA

        self.hydro_model = mariPower.ship.CBT()
        self.hydro_model.Draught = np.array([config.BOAT_DRAUGHT])
        self.hydro_model.Roughness_Level = np.array([config.BOAT_ROUGHNESS_LEVEL])
        self.hydro_model.Roughness_Distribution_Level = np.array([config.BOAT_ROUGHNESS_DISTRIBUTION_LEVEL])
        self.use_depth_data = True

    def set_ship_property(self, variable, value):
        print('Setting ship property ' + variable + ' to ' + str(value))
        setattr(self.hydro_model, variable, value)

    def print_init(self):
        logger.info(form.get_log_step('boat speed' + str(self.speed), 1))
        logger.info(form.get_log_step('path to weather data' + self.environment_path, 1))
        logger.info(form.get_log_step('path to CoursesRoute.nc' + self.courses_path, 1))
        logger.info(form.get_log_step('path to depth data' + self.depth_path, 1))

    def init_hydro_model_single_pars(self):
        self.hydro_model = mariPower.ship.CBT()
        # shipSpeed = 13 * 1852 / 3600
        self.hydro_model.WindDirection = math.radians(90)
        self.hydro_model.WindSpeed = 0
        self.hydro_model.TemperatureWater = 10

        self.hydro_model.WaveSignificantHeight = 2
        self.hydro_model.WavePeakPeriod = 10.0
        self.hydro_model.WaveDirection = math.radians(45)

        self.hydro_model.CurrentDirection = math.radians(0)
        self.hydro_model.CurrentSpeed = 0.5

        self.MaxIterations = 5

        logger.info('Setting environmental parameters of tanker:')
        logger.info('     water temp', self.hydro_model.TemperatureWater)
        logger.info('     wave significant height', self.hydro_model.WaveSignificantHeight)
        logger.info('     wave peak period', self.hydro_model.WavePeakPeriod)
        logger.info('     wave dir', self.hydro_model.WaveDirection)
        logger.info('     current dir', self.hydro_model.CurrentDirection)
        logger.info('     current speed', self.hydro_model.CurrentSpeed)

    #  initialise mariPower.ship for communication of courses via arrays and passing of environmental data as netCDF
    # def init_hydro_model_NetCDF(self, netCDF_filepath):
    #    self.hydro_model = mariPower.ship.CBT()
    #    self.environment_path = netCDF_filepath
    #    Fx, driftAngle, ptemp, n, delta = mariPower.__main__.PredictPowerForNetCDF(self.hydro_model, netCDF_filepath)

    def set_boat_speed(self, speed):
        self.speed = speed

    def set_env_data_path(self, path):
        self.environment_path = path

    def set_courses_path(self, path):
        self.courses_path = path

    ##
    # function that implements a dummy model for the estimation of the fuel consumption. Only to be used for code
    # testing, for which it minimises excecution time. Does not provide fully accurate estimations. Take care to
    # initialise the simple model using
    # calibrate_simple_fuel()
    def get_fuel_per_course_simple(self, course, wind_speed, wind_dir):
        debug = False
        angle = np.abs(course - wind_dir)
        if angle > 180:
            angle = np.abs(360 - angle)
        if debug:
            form.print_line()
            form.print_step('course = ' + str(course), 1)
            form.print_step('wind_speed = ' + str(wind_speed), 1)
            form.print_step('wind_dir = ' + str(wind_dir), 1)
            form.print_step('delta angle = ' + str(angle), 1)
        wind_speed = wind_speed
        power = self.simple_fuel_model.interp(delta_angle=angle, wind_speed=wind_speed)['power'].to_numpy()

        if debug:
            form.print_step('power = ' + str(power), 1)
        return power

    # def get_fuel_per_time_simple(self, delta_time):
    #    f = 0.0007 * self.rpm ** 3 + 0.0297 * self.rpm ** 2 + 2.8414 * self.rpm - 19.359  # fuel [kg/h]
    #    f *= delta_time / 3600 * 1 / 1000  # amount of fuel for this time interval
    #    return f

    ##
    # initiate estimation of power consumption in mariPower for one particular course and
    # wind direction and speed as well as boat speed
    def get_fuel_per_course(self, course, wind_dir, wind_speed, boat_speed):
        # boat_speed = np.array([boat_speed])
        self.hydro_model.WindDirection = math.radians(wind_dir)
        self.hydro_model.WindSpeed = wind_speed
        form.print_step('course [degrees]= ' + str(course), 1)
        course = units.degree_to_pmpi(course)
        form.print_step('course [rad]= ' + str(course), 1)
        form.print_step('wind dir = ' + str(self.hydro_model.WindDirection), 1)
        form.print_step('wind speed = ' + str(self.hydro_model.WindSpeed), 1)
        form.print_step('boat_speed = ' + str(boat_speed), 1)
        # Fx, driftAngle, ptemp, n, delta = self.hydro_model.IterateMotionSerial(course, boat_speed, aUseHeading=True,
        #                                                                 aUpdateCalmwaterResistanceEveryIteration=False)
        Fx, driftAngle, ptemp, n, delta = self.hydro_model.IterateMotion(course, boat_speed, aUseHeading=True,
                                                                         aUpdateCalmwaterResistanceEveryIteration=False)

        return ptemp

    ##
    # initialisation of simple fuel model that is used as dummy for accurate power estimation via mariPower
    def calibrate_simple_fuel(self):
        self.simple_fuel_model = xr.open_dataset(
            "/home/kdemmich/MariData/Code/MariGeoRoute/Isochrone/Data/SimpleFuelModel/simple_fuel_model.nc")
        form.print_line()
        logger.info('Initialising simple fuel model')
        logger.info(self.simple_fuel_model)

    ##
    # function to write a simple fuel model to file which can be used as dummy for the power estimation with
    # mariPower. The
    # model only considers wind speed and angle as well as the boat speed. 'n_angle' times 'n_wind_speed' pairs of
    # wind speed and wind angle
    # are generated and send to mariPower. The calculated power consumption and wind data are written to file and can
    # in the following be used as input for Tanker.calibrate_simple_fuel.
    def write_simple_fuel(self):
        n_angle = 10
        n_wind_speed = 20
        power = np.zeros((n_angle, n_wind_speed))
        delta_angle = units.get_bin_centers(0, 180, n_angle)
        wind_speed = units.get_bin_centers(0, 60, n_wind_speed)

        coords = dict(delta_angle=(["delta_angle"], delta_angle), wind_speed=(["wind_speed"], wind_speed), )
        attrs = dict(description="Necessary descriptions added here.")

        for iang in range(0, n_angle):
            for iwind_speed in range(0, n_wind_speed):
                course = 0
                wind_dir = 0 + delta_angle[iang]
                power[iang, iwind_speed] = self.get_fuel_per_course(course, wind_dir, wind_speed[iwind_speed],
                                                                    self.speed)

        data_vars = dict(power=(["delta_angle", "wind_speed"], power), )

        ds = xr.Dataset(data_vars, coords, attrs)
        ds.to_netcdf('/home/kdemmich/MariData/Code/simple_fuel_model.nc')

        logger.info('Writing simple fuel model:')
        logger.info(ds)

    ##
    # Initialise power estimation for a tuple of courses in dependence on wind speed and direction. The information
    # is send to mariPower per course.
    def get_fuel_per_time(self, courses, wind):
        debug = False

        # ToDo: use logger.debug and args.debug
        if (debug):
            print('Requesting power calculation')
            course_str = 'Courses:' + str(courses)
            form.print_step(course_str, 1)

        P = np.zeros(courses.shape)
        for icours in range(0, courses.shape[0]):
            # P[icours] = self.get_fuel_per_course(courses[icours], wind['twa'][icours], wind['tws'][icours],
            # self.speed)
            P[icours] = self.get_fuel_per_course_simple(courses[icours], wind['tws'][icours], wind['twa'][icours])
            if math.isnan(P[icours]):
                P[icours] = 1000000000000000

        if (debug):
            form.print_step('power consumption' + str(P))
        return P

    ##
    # Writes netCDF which stores courses in dependence on latitude, longitude and time for further processing by
    # mariPower.
    # Several courses can be provided per space point. In this case, the arrays lats and lons need to be filled
    # e.g. power estimation is requested for 3 courses (c1, c2, c3) for 1 space-time point (lat1, lon1) then:
    #   courses = {c1, c2, c3}
    #   lats = {lat1, lat1, lat1}
    #   lons = {lon1, lon1, lon1}

    def write_netCDF_courses(self, courses, lats, lons, time, speed=[], unique_coords=False):
        debug = False

        if speed == []:
            speed = np.repeat(self.speed, courses.shape, axis=0)

        courses = units.degree_to_pmpi(courses)

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

        ds.to_netcdf(self.courses_path + str())
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

        power = ds['Power_brake'].to_numpy().flatten() * u.Watt
        rpm = ds['RotationRate'].to_numpy().flatten() * u.Hz
        fuel = ds['Fuel_consumption_rate'].to_numpy().flatten() * u.tonne/u.hour
        fuel = fuel.to(u.kg/u.second)
        r_wind = ds['Wind_resistance'].to_numpy().flatten() * u.newton
        r_calm = ds['Calm_resistance'].to_numpy().flatten() * u.newton
        r_waves = ds['Wave_resistance'].to_numpy().flatten() * u.newton
        r_shallow = ds['Shallow_water_resistance'].to_numpy().flatten() * u.newton
        r_roughness = ds['Hull_roughness_resistance'].to_numpy().flatten() * u.newton
        speed = np.repeat(self.speed, power.shape)
        status = ds['Status'].to_numpy().flatten()

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
            status=status
        )

        if (debug):
            form.print_step('Dataset with fuel' + str(ds), 1)
            form.print_step('original shape power' + str(power.shape), 1)
            form.print_step('flattened shape power' + str(ship_params.get_power.shape), 1)
            form.print_step('power result' + str(ship_params.get_power))

        return ship_params

    ##
    # dummy function uses to mimic writing of power estimation to 'courses netCDF'. Only used for testing purposes.
    #
    def get_fuel_netCDF_dummy(self, ds, courses, wind):
        debug = False

        power = self.get_fuel_per_time(courses, wind)
        if (debug):
            form.print_step('power shape' + str(power.shape), 1)
        power = power.reshape(ds['lat'].shape[0], ds['it'].shape[0])
        ds["power"] = (['lat', 'it'], power)
        if (debug):
            form.print_step('power new shape' + str(power.shape), 1)
            form.print_step('ds' + str(ds), 1)

        ds.to_netcdf(self.courses_path)
        ds_read = xr.open_dataset(self.courses_path)
        # ToDo: use logger.debug and args.debug
        if (debug):
            print('read data set', ds_read)

    ##
    # Passes paths for 'courses netCDF' and 'environmental data netCDF' to mariPower and request estimation of power
    # consumption.
    # Is not yet working as explained for Tanker.get_fuel_netCDF_loop
    #
    def get_fuel_netCDF(self):
        mariPower_ship = copy.deepcopy(self.hydro_model)
        if self.use_depth_data:
            mariPower.__main__.PredictPowerOrSpeedRoute(mariPower_ship, self.courses_path, self.environment_path,
                                                        self.depth_path)
        else:
            mariPower.__main__.PredictPowerOrSpeedRoute(mariPower_ship, self.courses_path, self.environment_path)
        # form.print_current_time('time for mariPower request:', start_time)

        ds_read = xr.open_dataset(self.courses_path)
        return ds_read

    ##
    # @brief splits data in 'courses netCDF' one bunches per course per space point, sends them to mariPower
    # separately and merges them again afterwards.
    #
    # mariPower can currently handle only requests with 1 course per space point. Thus, the data in the 'course
    # netCDF' is split in
    # several bunchs each one containing an xarray with only one course per space point. The bunches are send to
    # mariPower separately
    # and the returned data sets are merged into one. Will (hopefully) be redundant as soon as mariPower accepts
    # requests with several
    # courses per space-time point and will then be replaced by Tanker.get_fuel_netCDF()
    def get_fuel_netCDF_loop(self):
        debug = False
        filename_single = '/home/kdemmich/MariData/Code/MariGeoRoute/Isochrone/CoursesRouteSingle.nc'
        # filename_single = 'C:/Users/Maneesha/Documents/GitHub/MariGeoRoute/WeatherRoutingTool/CoursesRouteSingle.nc'
        ds = xr.load_dataset(self.courses_path)
        n_vars = ds['it'].shape[0]
        ds_merged = xr.Dataset()

        if (debug):
            form.print_line()
            form.print_step('get_fuel_netCDF_loop: loop over all variants per space point', 0)
            form.print_step('original dataset: ' + str(ds), 0)

        for ivar in range(1, n_vars + 1):
            ds_read_temp = ds.isel(it=[ivar - 1])
            ds_read_temp.coords['it'] = [1]
            ds_read_temp.to_netcdf(filename_single, mode='w')
            ds_read_temp.close()
            ship = mariPower.ship.CBT()
            if (debug):
                ds_read_test = xr.load_dataset(filename_single)
                courses_test = ds_read_test['courses']
                form.print_step('courses_test' + str(courses_test.to_numpy()), 1)
                form.print_step('speed' + str(ds_read_test['speed'].to_numpy()), 1)
            # start_time = time.time()
            mariPower.__main__.PredictPowerOrSpeedRoute(ship, filename_single, self.environment_path, None, False,
                                                        False)
            # form.print_current_time('time for mariPower request:', start_time)

            ds_temp = xr.load_dataset(filename_single)
            ds_temp.coords['it'] = [ivar]
            if ivar == 1:
                ds_merged = ds_temp.copy()
            else:
                ds_merged = xr.concat([ds_merged, ds_temp], dim="it")
            if (debug):
                form.print_step('step ' + str(ivar) + ': merged dataset:' + str(ds_merged), 1)
        ds_merged['lon'] = ds_merged['lon'].sel(it=1).drop('it')
        ds_merged['time'] = ds_merged['time'].sel(it=1).drop('it')

        if (debug):
            form.print_step('final merged dataset:' + str(ds_merged))
        ds.close()
        return ds_merged

    ##
    # main function for communication with mariPower package (see documentation above)
    def get_ship_parameters(self, courses, lats, lons, time, speed=-99, unique_coords=False):
        self.write_netCDF_courses(courses, lats, lons, time, speed, unique_coords)

        # ds = self.get_fuel_netCDF_loop()
        # ds = self.get_fuel_netCDF_dummy(ds, courses, wind)
        ds = self.get_fuel_netCDF()
        ship_params = self.extract_params_from_netCDF(ds)
        ds.close()

        return ship_params

    ##
    # Function to test/plot power consumption in dependence of wind speed and direction. Works only with old versions
    # of mariPower package.
    # Has partly been replaced by test_polars: test_power_consumption_returned()
    def test_power_consumption_per_course(self):
        courses = np.linspace(0, 360, num=21, endpoint=True)
        wind_dir = 45
        wind_speed = 2
        power = np.zeros(courses.shape)

        # get_fuel_per_course gets angles in degrees from 0 to 360
        for i in range(0, courses.shape[0]):
            power[i] = self.get_fuel_per_course(courses[i], wind_dir, wind_speed,
                                                self.speed)  # power[i] = self.get_fuel_per_time_simple(i*3600)

        # plotting with matplotlib needs angles in radiants
        fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'})
        for i in range(0, courses.shape[0]):
            courses[i] = math.radians(courses[i])
        wind_dir = math.radians(wind_dir)

        axes[0].plot(courses, power)
        axes[0].legend()
        for ax in axes.flatten():
            ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
            ax.set_theta_zero_location("S")
            ax.grid(True)
        axes[1].plot([wind_dir, wind_dir], [0, wind_speed], color='blue', label='Wind')
        # axes[1].plot([self.hydro_model.WaveDirection, self.hydro_model.WaveDirection], [0,
        # 1 * (self.hydro_model.WaveSignificantHeight > 0.0)],
        #                color='green', label='Seaway')
        # axes[1].plot([self.hydro_model.CurrentDirection, self.hydro_model.CurrentDirection], [0,
        # 1 * (self.hydro_model.CurrentSpeed > 0.0)],
        #                color='purple', label='Current')
        axes[1].legend()

        axes[0].set_title("Power", va='bottom')
        axes[1].set_title("Environmental conditions", va='top')
        plt.show()

    ##
    # Function to test/plot power consumption in dependence of wind speed and direction. Works only with old versions
    # of mariPower package.
    # Has partly been replaced by test_polars: test_power_consumption_returned()
    def test_power_consumption_per_speed(self):
        course = 10
        boat_speed = np.linspace(1, 20, num=17)
        wind_dir = 45
        wind_speed = 2
        power = np.zeros(boat_speed.shape)

        for i in range(0, boat_speed.shape[0]):
            power[i] = self.get_fuel_per_course(course, wind_dir, wind_speed,
                                                boat_speed[i])  # power[i] = self.get_fuel_per_time_simple(i*3600)

        plt.plot(boat_speed, power, 'r--')
        plt.xlabel('speed (m/s)')
        plt.ylabel('power (W)')
        plt.show()
