"""Weather functions."""
import logging
import os
import time
from datetime import datetime, timedelta
from math import ceil

import datacube
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

import WeatherRoutingTool.utils.graphics as graphics
import WeatherRoutingTool.utils.formatting as form
from maridatadownloader import DownloaderFactory
from WeatherRoutingTool.utils.maps import Map
from WeatherRoutingTool.utils.unit_conversion import (check_dataset_spacetime_consistency, convert_nptd64_to_ints,
                                                      round_time)

logger = logging.getLogger('WRT.weather')

UNITS_DICT = {
    'Pressure_reduced_to_MSL_msl': ['Pa'],
    'so': ['1e-3'],
    'Temperature_surface': ['K'],
    'thetao': ['degrees_C', '°C'],
    'u-component_of_wind_height_above_ground': ['m s-1', 'm/s'],
    'v-component_of_wind_height_above_ground': ['m s-1', 'm/s'],
    'utotal': ['m s-1', 'm/s'],
    'vtotal': ['m s-1', 'm/s'],
    'VHM0': ['m'],
    'VMDR': ['degree'],
    'VTPK': ['s']
}


class WeatherCond:
    time_steps: int
    time_res: timedelta
    time_start: datetime
    time_end: timedelta
    map_size: Map
    ds: xr.Dataset

    def __init__(self, time, hours, time_res):
        form.print_line()
        logger.info('Initialising weather')

        self.time_res = time_res
        self.time_start = time
        self.time_end = time + timedelta(hours=hours)

        time_passed = self.time_end - self.time_start
        self.time_steps = int(time_passed.total_seconds() / self.time_res.total_seconds())

        logger.info(form.get_log_step('forecast from ' + str(self.time_start) + ' to ' + str(self.time_end), 1))
        logger.info(form.get_log_step('nof time steps ' + str(self.time_steps), 1))
        form.print_line()

    @property
    def time_res(self):
        return self._time_res

    @time_res.setter
    def time_res(self, value):
        if (value < 3):
            raise ValueError('Resolution below 3h not possible')
        self._time_res = timedelta(hours=value)
        logger.info(form.get_log_step('time resolution: ' + str(self._time_res) + ' hours', 1))

    @property
    def time_start(self):
        return self._time_start

    @time_start.setter
    def time_start(self, value):
        rounded_time = value - self.time_res / 2
        rounded_time = round_time(rounded_time, int(self.time_res.total_seconds()))
        self._time_start = rounded_time

    @property
    def time_end(self):
        return self._time_end

    @time_end.setter
    def time_end(self, value):
        rounded_time = value + self.time_res / 2
        rounded_time = round_time(rounded_time, int(self.time_res.total_seconds()))
        self._time_end = rounded_time

    def check_units(self):
        for var_name, data_array in self.ds.data_vars.items():
            if var_name in UNITS_DICT:
                if 'units' not in data_array.attrs:
                    logger.warning(f"Weather data variable '{var_name}' has no 'units' attribute.")
                else:
                    unit = data_array.attrs['units']
                    if unit not in UNITS_DICT[var_name]:
                        logger.warning(f"Weather data variable '{var_name}' has the wrong unit '{unit}', "
                                       f"should be one of '{UNITS_DICT[var_name]}'.")
            else:
                logger.warning(f"Weather data variable '{var_name}' found, but not expected. Will be ignored.")

    def set_map_size(self, map):
        self.map_size = map

    def get_map_size(self):
        return self.map_size

    def read_dataset(self, filepath=None):
        pass


class WeatherCondEnvAutomatic(WeatherCond):

    # FIXME: add currents?

    def __init__(self, time, hours, time_res):
        super().__init__(time, hours, time_res)

    def check_data_consistency(self, ds_CMEMS_phys, ds_CMEMS_wave, ds_CMEMS_curr, ds_GFS):
        ############################################
        # check time consistency

        # check time resolution and shifts
        check_dataset_spacetime_consistency(ds_GFS, ds_CMEMS_phys, 'time', 'GFS', 'CMEMS physics')
        check_dataset_spacetime_consistency(ds_GFS, ds_CMEMS_wave, 'time', 'GFS', 'CMEMS waves')
        check_dataset_spacetime_consistency(ds_GFS, ds_CMEMS_curr, 'time', 'GFS', 'CMEMS currents')

        # hard asserts in case situation changes with respect to expected behaviour
        time_wave = ds_CMEMS_wave['time'].to_numpy()
        time_wind = ds_CMEMS_phys['time'].to_numpy()
        time_curr = ds_CMEMS_curr['time'].to_numpy()
        time_GFS = ds_GFS['time'].to_numpy()

        time_wave_sec = np.full(time_wave.shape[0], 0)
        time_wind_sec = np.full(time_wind.shape[0], 0)
        time_curr_sec = np.full(time_curr.shape[0], 0)
        time_GFS_sec = np.full(time_GFS.shape[0], 0)

        assert time_wave.shape[0] + 1 == time_wind.shape[
            0]  # CMEMS wave dataset contains 1 more time step than CMEMS physics
        assert time_wave.shape[0] + 1 == time_curr.shape[
            0]  # CMEMS current dataset contains 1 more time step than CMEMS physics
        assert time_wave.shape[0] == time_GFS.shape[0]

        for itime in range(0, time_wave.shape[0]):
            time_wave_sec[itime] = convert_nptd64_to_ints(time_wave[itime])
        time_wave_sec = time_wave_sec - 30 * 60
        time_wave_sec = np.append(time_wave_sec, time_wave_sec[time_wave_sec.shape[0] - 1] + 3 * 60 * 60)
        for itime in range(0, time_wind.shape[0]):
            time_wind_sec[itime] = convert_nptd64_to_ints(time_wind[itime])
        for itime in range(0, time_curr.shape[0]):
            time_curr_sec[itime] = convert_nptd64_to_ints(time_curr[itime])
        for itime in range(0, time_GFS.shape[0]):
            time_GFS_sec[itime] = convert_nptd64_to_ints(time_GFS[itime])
        time_GFS_sec = time_GFS_sec - 30 * 60
        time_GFS_sec = np.append(time_GFS_sec, time_GFS_sec[time_GFS_sec.shape[0] - 1] + 3 * 60 * 60)

        assert np.array_equal(time_wind_sec, time_wave_sec)
        assert np.array_equal(time_wind_sec, time_GFS_sec)
        assert np.array_equal(time_wind_sec, time_curr_sec)

        ############################################
        # check space consistency

        check_dataset_spacetime_consistency(ds_GFS, ds_CMEMS_phys, 'latitude', 'GFS', 'CMEMS physics')
        check_dataset_spacetime_consistency(ds_GFS, ds_CMEMS_wave, 'latitude', 'GFS', 'CMEMS waves')
        check_dataset_spacetime_consistency(ds_GFS, ds_CMEMS_curr, 'latitude', 'GFS', 'CMEMS currents')

        check_dataset_spacetime_consistency(ds_GFS, ds_CMEMS_phys, 'longitude', 'GFS', 'CMEMS physics')
        check_dataset_spacetime_consistency(ds_GFS, ds_CMEMS_wave, 'longitude', 'GFS', 'CMEMS waves')
        check_dataset_spacetime_consistency(ds_GFS, ds_CMEMS_curr, 'longitude', 'GFS', 'CMEMS currents')

    def read_dataset(self, filepath=None):
        CMEMS_product_wave = 'cmems_mod_glo_wav_anfc_0.083deg_PT3H-i'
        CMEMS_product_wind = 'cmems_mod_glo_phy_anfc_0.083deg_PT1H-m'
        logger.info(form.get_log_step(
            'Loading datasets from GFS and CMEMS (' + CMEMS_product_wind + ' and ' + CMEMS_product_wave, 1))

        time_min = self.time_start.strftime("%Y-%m-%dT%H:%M:%S")
        time_max = self.time_end.strftime("%Y-%m-%dT%H:%M:%S")

        time_min_CMEMS_phys = (self.time_start - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%S")
        time_max_CMEMS_phys = (self.time_end + timedelta(minutes=180)).strftime("%Y-%m-%dT%H:%M:%S")

        lon_min = self.map_size.lon1
        lon_max = self.map_size.lon2
        lat_min = self.map_size.lat1
        lat_max = self.map_size.lat2
        height_min = 10
        height_max = 20

        cmems_username = os.getenv('CMEMS_USERNAME')
        cmems_password = os.getenv('CMEMS_PASSWORD')

        # download GFS data
        par_GFS = ["Temperature_surface", "u-component_of_wind_height_above_ground",
                   "v-component_of_wind_height_above_ground", "Pressure_reduced_to_MSL_msl"]
        sel_dict_GFS = {'time': slice(time_min, time_max), 'time1': slice(time_min, time_max),
                        'height_above_ground2': slice(height_min, height_max), 'longitude': slice(lon_min, lon_max),
                        'latitude': slice(lat_min, lat_max)}

        downloader_gfs = DownloaderFactory.get_downloader('opendap', 'gfs')
        ds_GFS = downloader_gfs.download(par_GFS, sel_dict_GFS)

        # download CMEMS wave data
        par_CMEMS_wave = ["VMDR", "VHM0", "VTPK"]
        sel_dict_CMEMS_wave = {'time': slice(time_min, time_max), 'latitude': slice(lat_min, lat_max),
                               'longitude': slice(lon_min, lon_max)}
        downloader_cmems_wave = DownloaderFactory.get_downloader(downloader_type='opendap', platform='cmems',
                                                                 product='cmems_mod_glo_wav_anfc_0.083deg_PT3H-i',
                                                                 product_type='nrt', username=cmems_username,
                                                                 password=cmems_password)
        ds_CMEMS_wave = downloader_cmems_wave.download(parameters=par_CMEMS_wave, sel_dict=sel_dict_CMEMS_wave)

        # download CMEMS physics data
        par_CMEMS_phys = ["thetao", "so"]
        sel_dict_CMEMS_phys = {'time': slice(time_min_CMEMS_phys, time_max_CMEMS_phys, 3),
                               'latitude': slice(lat_min, lat_max), 'longitude': slice(lon_min, lon_max)}
        downloader_cmems_phys = DownloaderFactory.get_downloader(downloader_type='opendap', platform='cmems',
                                                                 product='cmems_mod_glo_phy_anfc_0.083deg_PT1H-m',
                                                                 product_type='nrt', username=cmems_username,
                                                                 password=cmems_password)
        ds_CMEMS_phys = downloader_cmems_phys.download(parameters=par_CMEMS_phys, sel_dict=sel_dict_CMEMS_phys)

        # download CMEMS current data
        par_CMEMS_curr = ["vtotal", "utotal"]
        sel_dict_CMEMS_curr = {'time': slice(time_min_CMEMS_phys, time_max_CMEMS_phys, 3),
                               'latitude': slice(lat_min, lat_max), 'longitude': slice(lon_min, lon_max)}
        downloader_cmems_curr = DownloaderFactory.get_downloader(downloader_type='opendap', platform='cmems',
                                                                 product='cmems_mod_glo_phy_anfc_merged-uv_PT1H-i',
                                                                 product_type='nrt', username=cmems_username,
                                                                 password=cmems_password)
        ds_CMEMS_curr = downloader_cmems_curr.download(parameters=par_CMEMS_curr, sel_dict=sel_dict_CMEMS_curr)

        # convert latitudes of GFS data
        GFS_lat = ds_GFS['latitude'].to_numpy()
        GFS_lat[GFS_lat < 0] = GFS_lat[GFS_lat < 0] + 180

        form.print_current_time('weather request:', time.time())
        self.check_data_consistency(ds_CMEMS_phys, ds_CMEMS_wave, ds_CMEMS_curr, ds_GFS)

        # interpolate CMEMS wave data to timestamps of CMEMS physics and merge
        phys_interpolated = ds_CMEMS_phys.interp_like(ds_CMEMS_wave)
        curr_interpolated = ds_CMEMS_curr.interp_like(ds_CMEMS_wave)
        full_CMEMS_data = xr.merge([curr_interpolated, phys_interpolated, ds_CMEMS_wave])
        form.print_current_time('CMEMS merge', time.time())

        # interpolate GFS data to lat/lon resolution of CMEMS full data and merge
        check_dataset_spacetime_consistency(ds_GFS, full_CMEMS_data, 'latitude', 'GFS', 'Full CMEMS')
        check_dataset_spacetime_consistency(ds_GFS, full_CMEMS_data, 'longitude', 'GFS', 'Full CMEMS')
        check_dataset_spacetime_consistency(ds_GFS, full_CMEMS_data, 'time', 'GFS', 'Full CMEMS')

        GFS_interpolated = ds_GFS.interp_like(full_CMEMS_data)
        form.print_current_time('interpolation', time.time())
        self.ds = xr.merge([full_CMEMS_data, GFS_interpolated])
        form.print_current_time('end time', time.time())

    def write_data(self, filepath):
        # time_str_start = self.time_start.strftime("%Y-%m-%d-%H")
        # time_str_end = self.time_end.strftime("%Y-%m-%d-%H")

        # filename = str(time_str_start) + '_' + str(time_str_end) + '_' + str(self.map_size.lat1) + '_' + str(
        #    self.map_size.lon1) + '_' + str(self.map_size.lat2) + '_' + str(self.map_size.lon2) + '.nc'
        # full_path = filepath + '/' + filename
        logger.info('Writing weather data to file ' + str(filepath))
        self.ds.to_netcdf(filepath)
        self.ds.close()
        return filepath


class WeatherCondFromFile(WeatherCond):
    wind_functions: None
    wind_vectors: None

    def __init__(self, time, hours, time_res):
        super().__init__(time, hours, time_res)

    def calculate_wind_function(self, time):
        time_str = time.strftime('%Y-%m-%d %H:%M:%S')
        # print('Reading time', time_str)

        try:
            u = self.ds['u-component_of_wind_height_above_ground'].sel(time=time_str, height_above_ground2=10)
            v = self.ds['v-component_of_wind_height_above_ground'].sel(time=time_str, height_above_ground2=10)
        except KeyError:
            time = self.ds['time']
            logger.error('time: ', time.to_numpy())
            logger.error('time string: ', time_str)
            raise Exception(
                'Please make sure that time stamps of environmental data match full hours: time = ' + time_str)

        twa, tws = self.get_twatws_from_uv(u, v)

        tws = tws.to_numpy()
        twa = twa.to_numpy()

        return {'twa': twa, 'tws': tws}

    def read_wind_functions(self, iTime):
        time = self.time_start + self.time_res * iTime
        # wind = self.nc_to_wind_function_old_format()
        wind = self.calculate_wind_function(time)

        if not (wind['twa'].shape == wind['tws'].shape):
            raise ValueError('Shape of twa and tws not matching!')

        lat_shape = wind['twa'].shape[0]
        lon_shape = wind['twa'].shape[1]
        lats_grid = np.linspace(self.map_size.lat1, self.map_size.lat2, lat_shape)
        lons_grid = np.linspace(self.map_size.lon1, self.map_size.lon2, lon_shape)

        f_twa = RegularGridInterpolator((lats_grid, lons_grid), wind['twa'], )

        f_tws = RegularGridInterpolator((lats_grid, lons_grid), wind['tws'], )

        return {'twa': f_twa, 'tws': f_tws, 'timestamp': time}

    def read_wind_vectors(self, time):
        """Return u-v components for given rect for visualization."""

        lat1 = self.map_size.lat1
        lat2 = self.map_size.lat2
        lon1 = self.map_size.lon1
        lon2 = self.map_size.lon2
        time_str = time.strftime('%Y-%m-%d %H:%M:%S')

        ds_time = self.ds.sel(time=time_str)

        u = ds_time['u-component_of_wind_height_above_ground'].where(
            (ds_time.latitude >= lat1) & (ds_time.latitude <= lat2) & (ds_time.longitude >= lon1) & (
                    ds_time.longitude <= lon2) & (ds_time.height_above_ground2 == 10), drop=True)
        v = ds_time['v-component_of_wind_height_above_ground'].where(
            (ds_time.latitude >= lat1) & (ds_time.latitude <= lat2) & (ds_time.longitude >= lon1) & (
                    ds_time.longitude <= lon2) & (ds_time.height_above_ground2 == 10), drop=True)
        lats_u_1D = ds_time['latitude'].where((ds_time.latitude >= lat1) & (ds_time.latitude <= lat2), drop=True)
        lons_u_1D = ds_time['longitude'].where((ds_time.longitude >= lon1) & (ds_time.longitude <= lon2), drop=True)

        u = u.to_numpy()
        v = v.to_numpy()
        lats_u_1D = lats_u_1D.to_numpy()
        lons_u_1D = lons_u_1D.to_numpy()
        lats_u = np.tile(lats_u_1D[:, np.newaxis], u.shape[1])
        lons_u = np.tile(lons_u_1D, (u.shape[0], 1))

        return {'u': u, 'v': v, 'lats_u': lats_u, 'lons_u': lons_u, 'timestamp': time}

    def plot_weather_map(self, fig, ax, time, varname):
        rebinx = 5
        rebiny = 5

        if varname == 'wind':
            u = self.ds['u-component_of_wind_height_above_ground'].where(self.ds.VHM0 > 0).sel(
                time=time,
                height_above_ground=10,
                latitude=slice(self.map_size.lat1, self.map_size.lat2),
                longitude=slice(self.map_size.lon1, self.map_size.lon2)
            )
            v = self.ds['v-component_of_wind_height_above_ground'].where(self.ds.VHM0 > 0).sel(
                time=time,
                height_above_ground=10,
                latitude=slice(self.map_size.lat1, self.map_size.lat2),
                longitude=slice(self.map_size.lon1, self.map_size.lon2)
            )

            u = u.coarsen(latitude=rebinx, longitude=rebiny, boundary="trim").mean()
            v = v.coarsen(latitude=rebinx, longitude=rebiny, boundary="trim").mean()

            windspeed = np.sqrt(u ** 2 + v ** 2)

            cp = windspeed.plot(alpha=0.5)
            # cp.set_clim(0, 20)
            plt.title('wind speed and direction')
            plt.rcParams['font.size'] = '20'
            plt.title('current')
            plt.ylabel('latitude (°N)', fontsize=20)
            plt.xlabel('longitude (°W)', fontsize=20)
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(20)
            x = windspeed.coords['longitude'].values
            y = windspeed.coords['latitude'].values
            # plt.quiver(x, y, u.values, v.values, clim=[0, 20])
            plt.barbs(x, y, u.values, v.values, clim=[0, 20])

        if varname == 'waveheight':
            height = self.ds['VHM0'].sel(time=time)
            h = height.plot()
            h.set_clim(0, 7)
            plt.title('wave heigh')
            plt.rcParams['font.size'] = '20'
            plt.title('current')
            plt.ylabel('latitude (°N)', fontsize=20)
            plt.xlabel('longitude (°W)', fontsize=20)
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(20)
            plt.show()

        if varname == 'current':
            u = self.ds['uo'].sel(time=time)
            v = self.ds['vo'].sel(time=time)
            u = u.isel(depth=0)
            v = v.isel(depth=0)

            # u = u[::3, ::6]
            # v = u[::3, ::6]

            unp = u.to_numpy()
            vnp = v.to_numpy()
            unp = graphics.rebin(unp, rebinx, rebiny)
            vnp = graphics.rebin(vnp, rebinx, rebiny)

            windspeed = np.sqrt(u ** 2 + v ** 2)
            c = windspeed.plot()
            c.set_clim(0, 0.6)
            plt.rcParams['font.size'] = '20'
            plt.title('current')
            plt.ylabel('latitude (°N)', fontsize=20)
            plt.xlabel('longitude (°W)', fontsize=20)
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(20)
            x = windspeed.coords['longitude'].values
            y = windspeed.coords['latitude'].values

            # plt.barbs(x, y, u.values, v.values)
            plt.quiver(x, y, u.values, v.values)
            plt.show()

        if varname == 'wavedir':
            wavedir = self.ds['VMDR'].sel(time=time)
            waveheight = self.ds['VHM0'].sel(time=time)

            u = np.cos(wavedir) * waveheight
            v = np.sin(wavedir) * waveheight

            h = waveheight.plot()
            # h.set_clim(0, 0.6)
            plt.rcParams['font.size'] = '20'
            plt.title('current')
            plt.ylabel('latitude (°N)', fontsize=20)
            plt.xlabel('longitude (°W)', fontsize=20)
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(20)
            x = waveheight.coords['longitude'].values
            y = waveheight.coords['latitude'].values

            # plt.barbs(x, y, u.values, v.values)
            plt.quiver(x, y, u.values, v.values)
            plt.show()

    def close_env_file(self):
        self.ds.close()

    def adjust_depth_format(self, depth_path):
        debug = True
        ds_depth = xr.open_dataset(depth_path)
        ds_depth = ds_depth.sortby("latitude")
        ds_depth.load()
        ds_depth_pos = ds_depth.where(ds_depth.longitude <= 180, drop=True)
        ds_depth_neg = ds_depth.where(ds_depth.longitude > 180, drop=True)
        ds_depth_neg['longitude'] = ds_depth_neg['longitude'] - 360
        ds_depth = ds_depth_pos.merge(ds_depth_neg)

        # ToDo: use logger.debug and args.debug
        if (debug):
            print('ds_depth_pos', ds_depth_pos)
            print('ds_depth_neg', ds_depth_neg)
            print('ds_depth new', ds_depth)

        return ds_depth

    def add_depth_to_EnvData(self, depth_path, bWriteEnvData=False):
        try:
            lat_start = self.map_size.lat1
            lat_end = self.map_size.lat2
            lon_start = self.map_size.lon1
            lon_end = self.map_size.lon2
        except ValueError:
            raise Exception('Need to initialise weather data bounding box before adding depth data!')

        ds_depth = xr.open_dataset(depth_path)
        ds_depth = ds_depth.where((ds_depth.lat > lat_start) & (ds_depth.lat < lat_end) & (ds_depth.lon > lon_start) & (
                ds_depth.lon < lon_end) & (ds_depth.z < 0), drop=True)

        ds_depth = ds_depth.rename(lat="latitude", lon="longitude")
        weather_int = self.ds.interp_like(ds_depth, method="linear")

        depth = ds_depth['z'].to_numpy()
        depth = np.nan_to_num(depth)

        weather_int['z'] = (['latitude', 'longitude'], depth)
        depth_test = weather_int['z'].to_numpy()
        if np.isnan(depth_test).any():
            logger.error('depth_test:', depth_test)
            raise Exception('element of depth is nan!')
        self.ds = weather_int

        if bWriteEnvData:
            self.ds.to_netcdf(
                '/home/kdemmich/MariData/Code/MariGeoRoute/Isochrone/Data/Depth_u_EnvData/EnvData_Depth.nc')

    def get_time_step_index(self, time):
        rounded_time = round_time(time, int(self.time_res.total_seconds()))
        time_passed = rounded_time - self.time_start
        idx = (time_passed.total_seconds() / self.time_res.total_seconds())
        return {'rounded_time': rounded_time, 'idx': idx}

    def manipulate_dataset(self):
        # condition =  4
        # lat = 54.608889
        # lon = 6.179722
        condition = 8
        lat = 55.048333
        lon = 5.130000

        # condition = 4
        # condition = 8

        dim = 0.25
        xmin = lon - dim
        xmax = lon + dim
        ymin = lat - dim
        ymax = lat + dim
        ll = dict(longitude=slice(xmin, xmax), latitude=slice(ymin, ymax))
        logger.info('before: ', self.ds["VHM0"].loc[ll].to_numpy())
        self.ds["VHM0"].loc[ll] = condition
        logger.info('after: ', self.ds["VHM0"].loc[ll].to_numpy())
        self.ds.to_netcdf('/home/kdemmich/MariData/Simulationsstudie_April23/manipulated_data.nc')

        return self.ds

    def get_twatws_from_uv(self, u, v):
        tws = np.sqrt(u ** 2 + v ** 2)
        twa = 180.0 / np.pi * np.arctan2(u, v) + 180.0  # angle from 0° to 360°, 0° = N
        return twa, tws

    def init_wind_vectors(self):
        """Return wind vectors for given number of hours.
            Parameters:
                    model (dict): available forecast wind functions
                    hours_ahead (int): number of hours looking ahead
                    lats, lons: rectange defining forecast area
            Returns:
                    wind_vectors (dict):
                        model: model timestamp
                        hour: function for given forecast hour
            """

        wind_vectors = {}
        wind_vectors['start_time'] = self.time_start

        for i in range(self.time_steps):
            time = self.time_start + self.time_res * i
            wind_vectors[i] = self.read_wind_vectors(time)  # print('reading wind vector time', time)

        self.wind_vectors = wind_vectors

    def get_wind_vector(self, time):
        time_passed = self.get_time_step_index(time)
        rounded_time = time_passed['rounded_time']
        idx = time_passed['idx']

        try:
            wind_timestamp = self.wind_vectors[idx]['timestamp']
        except KeyError:
            logger.error('Requesting weather data for ' + str(time) + ' at index ' + str(idx) + ' but only ' + str(
                self.time_steps) + ' available')
            raise

        if not (rounded_time == wind_timestamp):
            ex = 'Accessing wrong weather forecast. Accessing element ' + str(
                self.wind_vectors[idx]['timestamp']) + ' but current rounded time is ' + str(rounded_time)
            raise Exception(ex)

        return self.wind_vectors[idx]

    def read_dataset(self, filepath=None):
        if filepath is None:
            raise RuntimeError("filepath must not be None for data_mode = 'from_file'")
        logger.info(form.get_log_step('Reading dataset from' + str(filepath), 1))
        self.ds = xr.open_dataset(filepath)  # self.ds = self.manipulate_dataset()


class WeatherCondODC(WeatherCond):
    def __init__(self, time, hours, time_res):
        super().__init__(time, hours, time_res)
        self.dc = datacube.Datacube()

    def load_odc_product(self, product_name, res_x, res_y, output_crs="EPSG:4326", measurements=None):
        try:
            if product_name not in list(self.dc.list_products().index):
                raise ValueError(f"{product_name} is not known in the Open Data Cube instance")

            time_min = self.time_start.strftime("%Y-%m-%dT%H:%M:%S")
            time_max = self.time_end.strftime("%Y-%m-%dT%H:%M:%S")

            lon_min = self.map_size.lon1
            lon_max = self.map_size.lon2
            lat_min = self.map_size.lat1
            lat_max = self.map_size.lat2

            if measurements is None:
                measurements = list(self.dc.list_measurements().loc[product_name].index)
            else:
                # Check if requested measurements are available in ODC (measurements or aliases)
                measurements_odc = list(self.dc.list_measurements().loc[product_name].index)
                aliases_odc = [alias for aliases_per_var in
                               list(self.dc.list_measurements().loc[product_name]['aliases']) for alias in
                               aliases_per_var]
                for measurement in measurements:
                    if (measurement not in measurements_odc) and (measurement not in aliases_odc):
                        raise KeyError(f"{measurement} is not a valid measurement for odc product {product_name}")
            # FIXME: is the order (res_x, res_y) correct in resolution and align?
            # FIXME: do we need a minus sign for res_x?
            query = {'resolution': (res_x, res_y), 'align': (res_x / 2, res_y / 2), 'latitude': (lat_min, lat_max),
                     'longitude': (lon_min, lon_max), 'output_crs': output_crs, 'time': (time_min, time_max),
                     'measurements': measurements}
            ds_datacube = self.dc.load(product=product_name, **query)
            # Apply scale_factor and offset if necessary (needs to be done explicitly as ODC is only setting
            # the attributes)
            if self._has_scaling(ds_datacube):
                ds_datacube = self._scale(ds_datacube)
            return ds_datacube
        except Exception as e:
            raise e

    def read_dataset(self, filepath=None):
        # ODC doesn't allow hyphens ("-") in band names. Because we would like to keep the original band
        # names from GFS with hyphen we use band aliases instead.
        measurements_gfs = ['Temperature_surface', 'Pressure_reduced_to_MSL_msl', 'Wind_speed_gust_surface',
                            'u-component_of_wind_height_above_ground', 'v-component_of_wind_height_above_ground']

        ds_CMEMS_phys = self.load_odc_product('physics', res_x=1 / 12, res_y=1 / 12)
        ds_CMEMS_wave = self.load_odc_product('waves', res_x=1 / 12, res_y=1 / 12)
        ds_CMEMS_curr = self.load_odc_product('currents', res_x=1 / 12, res_y=1 / 12)
        ds_GFS = self.load_odc_product('weather', res_x=0.25, res_y=0.25, measurements=measurements_gfs)

        # form.print_current_time('time after weather request:', time.time())
        # self.check_data_consistency(ds_CMEMS_phys, ds_CMEMS_wave, ds_GFS)
        form.print_current_time('weather checks:', time.time())
        # interpolate CMEMS wave data to timestamps of CMEMS physics and merge
        phys_interpolated = ds_CMEMS_phys.interp_like(ds_CMEMS_wave)
        curr_interpolated = ds_CMEMS_curr.interp_like(ds_CMEMS_wave)
        full_CMEMS_data = xr.merge([curr_interpolated, phys_interpolated, ds_CMEMS_wave])
        form.print_current_time('CMEMS merge', time.time())
        # interpolate GFS data to lat/lon resolution of CMEMS full data and merge
        check_dataset_spacetime_consistency(ds_GFS, full_CMEMS_data, 'latitude', 'GFS', 'Full CMEMS')
        check_dataset_spacetime_consistency(ds_GFS, full_CMEMS_data, 'longitude', 'GFS', 'Full CMEMS')
        check_dataset_spacetime_consistency(ds_GFS, full_CMEMS_data, 'time', 'GFS', 'Full CMEMS')

        GFS_interpolated = ds_GFS.interp_like(full_CMEMS_data)
        form.print_current_time('interpolation', time.time())
        self.ds = xr.merge([full_CMEMS_data, GFS_interpolated])
        form.print_current_time('end time', time.time())

    def write_data(self, filepath):
        logger.info('Writing weather data to file ' + str(filepath))
        self.ds.to_netcdf(filepath)
        self.ds.close()
        return filepath

    def _has_scaling(self, dataset):
        """Check if any of the included data variables has a scale_factor or add_offset"""
        for var in dataset.data_vars:
            if 'scale_factor' in dataset[var].attrs or 'add_offset' in dataset[var].attrs:
                return True
        return False

    def _scale(self, dataset):
        # FIXME: decode_cf also scales the nodata values, e.g. -32767 -> -327.67
        return xr.decode_cf(dataset)


class FakeWeather(WeatherCond):
    def __init__(self, time, hours, time_res, coord_res=1/12, var_dict=None):
        super().__init__(time, hours, time_res)
        self.var_dict = {}
        var_list_zero = {
            'vtotal': 0, 'utotal': 0,
            'thetao': 0,
            'so': 0,
            'VMDR': 0, 'VHM0': 0, 'VTPK': 0,
            'Temperature_surface': 0,
            'u-component_of_wind_height_above_ground': 0, 'v-component_of_wind_height_above_ground': 0,
            'Pressure_reduced_to_MSL_msl': 0
        }
        self.coord_res = coord_res

        self.combine_var_dicts(var_dict_manual=var_dict, var_dict_zero=var_list_zero)

    def combine_var_dicts(self, var_dict_manual, var_dict_zero):
        if var_dict_manual:
            self.var_dict = var_dict_zero | var_dict_manual
        else:
            self.var_dict = var_dict_zero

    def read_dataset(self, filepath=None):
        # initialise coordinates
        # round differences to 10^-5 (~1 m for latitude) to prevent shifts due to floating numbers
        # interpret the configured map size limits inclusive
        n_lat_values = ceil(round(self.map_size.lat2 - self.map_size.lat1, 5)/self.coord_res) + 1
        lat_start = self.map_size.lat1
        lat_end = self.map_size.lat1 + self.coord_res * (n_lat_values-1)
        lat = np.linspace(lat_start, lat_end, n_lat_values)

        n_lon_values = ceil(round(self.map_size.lon2 - self.map_size.lon1, 5)/self.coord_res) + 1
        lon_start = self.map_size.lon1
        lon_end = self.map_size.lon1 + self.coord_res * (n_lon_values-1)
        lon = np.linspace(lon_start, lon_end, n_lon_values)

        n_time_values = self.time_steps + 1
        start_time_sec = self.time_start.timestamp()
        end_time_sec = start_time_sec + self.time_steps * self.time_res.total_seconds()
        time_space = np.linspace(start_time_sec, end_time_sec, n_time_values)
        time = np.full(time_space.shape[0], datetime.today())
        for i in range(0, n_time_values):
            time[i] = datetime.fromtimestamp(time_space[i])

        n_depth_values = 1
        depth = np.array([0.494])

        # initialise variables
        vtotal = np.full((n_lat_values, n_lon_values, n_time_values, n_depth_values), self.var_dict['vtotal'])
        utotal = np.full((n_lat_values, n_lon_values, n_time_values, n_depth_values), self.var_dict['utotal'])
        thetao = np.full((n_lat_values, n_lon_values, n_time_values, n_depth_values), self.var_dict['thetao'])
        so = np.full((n_lat_values, n_lon_values, n_time_values, n_depth_values), self.var_dict['so'])

        VMDR = np.full((n_lat_values, n_lon_values, n_time_values), self.var_dict['VMDR'])
        VHM0 = np.full((n_lat_values, n_lon_values, n_time_values), self.var_dict['VHM0'])
        VTPK = np.full((n_lat_values, n_lon_values, n_time_values), self.var_dict['VTPK'])
        Temperature_surface = np.full((n_lat_values, n_lon_values, n_time_values),
                                      self.var_dict['Temperature_surface'])

        uwind = np.full((n_lat_values, n_lon_values, n_time_values),
                        self.var_dict['u-component_of_wind_height_above_ground'])
        vwind = np.full((n_lat_values, n_lon_values, n_time_values),
                        self.var_dict['v-component_of_wind_height_above_ground'])

        Pressure_reduced_to_MSL_msl = np.full((n_lat_values, n_lon_values, n_time_values),
                                              self.var_dict['Pressure_reduced_to_MSL_msl'])

        # create dataset
        data_vars = dict(vtotal=(["latitude", "longitude", "time", "depth"], vtotal),
                         utotal=(["latitude", "longitude", "time", "depth"], utotal),
                         thetao=(["latitude", "longitude", "time", "depth"], thetao),
                         so=(["latitude", "longitude", "time", "depth"], so),

                         VMDR=(["latitude", "longitude", "time"], VMDR),
                         VHM0=(["latitude", "longitude", "time"], VHM0),
                         VTPK=(["latitude", "longitude", "time"], VTPK),
                         Temperature_surface=(["latitude", "longitude", "time"], Temperature_surface),

                         uwind=(["latitude", "longitude", "time"], uwind),
                         vwind=(["latitude", "longitude", "time"], vwind),

                         Pressure_reduced_to_MSL_msl=(["latitude", "longitude", "time"], Pressure_reduced_to_MSL_msl),
                         )
        coords = dict(latitude=(["latitude"], lat), longitude=(["longitude"], lon), time=(["time"], time),
                      depth=(["depth"], depth))
        attrs = dict(description="Necessary descriptions added here.")

        ds = xr.Dataset(data_vars, coords, attrs)

        ds['vtotal'] = ds['vtotal'].assign_attrs(units='m s-1')
        ds['utotal'] = ds['utotal'].assign_attrs(units='m s-1')
        ds['thetao'] = ds['thetao'].assign_attrs(units='degrees_C')
        ds['so'] = ds['so'].assign_attrs(units='1e-3')
        ds['VMDR'] = ds['VMDR'].assign_attrs(units='degree')
        ds['VHM0'] = ds['VHM0'].assign_attrs(units='m')
        ds['VTPK'] = ds['VTPK'].assign_attrs(units='s')
        ds['Temperature_surface'] = ds['Temperature_surface'].assign_attrs(units='K')
        ds['uwind'] = ds['uwind'].assign_attrs(units='m/s')
        ds['vwind'] = ds['vwind'].assign_attrs(units='m/s')
        ds['Pressure_reduced_to_MSL_msl'] = ds['Pressure_reduced_to_MSL_msl'].assign_attrs(units='Pa')

        ds = ds.rename({'uwind': 'u-component_of_wind_height_above_ground',
                        'vwind': 'v-component_of_wind_height_above_ground'})

        self.ds = ds

    def write_data(self, filepath):
        logger.info('Writing weather data to file ' + str(filepath))
        self.ds.to_netcdf(filepath)
        self.ds.close()
        return filepath
