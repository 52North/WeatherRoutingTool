import logging
from pathlib import Path

import numpy as np
import xarray as xr
from astropy import units as u

import WeatherRoutingTool.utils.formatting as form
from WeatherRoutingTool.ship.shipparams import ShipParams
from WeatherRoutingTool.ship.ship_config import ShipConfig

logger = logging.getLogger('WRT.ship')


# Boat: Main class for boats. Classes 'Tanker' and 'SailingBoat' derive from it
# Tanker: implements interface to mariPower package which is used for power estimation.

class Boat:
    """
    Base class representing a vessel used by the Weather Routing Tool.

    - Sub-classes (for example `MariPowerTanker`, `DirectPowerBoat` or `SailingBoat`) implement
        vessel-specific power and fuel estimation methods. This class provides shared
        functionality used by all boat types such as keeping basic ship parameters
        (draught, under-keel clearance), handling weather data access and helper
        methods for extracting weather fields from a NetCDF file.
    - Sub-classes should implement vessel-specific behaviour by overriding
        ``get_ship_parameters``. Individual methods are documented on the method
        itself.

    :param weather_path: Path to the NetCDF file that contains
        weather/oceanographic variables.
    :type weather_path: str
    :param under_keel_clearance: Minimum clearance below the hull as defined in
        the ship configuration.
    :type under_keel_clearance: astropy.units.Quantity
    :param draught_aft: Aft draught read from the ship configuration.
    :type draught_aft: astropy.units.Quantity
    :param draught_fore: Fore draught read from the ship configuration.
    :type draught_fore: astropy.units.Quantity
    :param time_min_max: Cached min/max time values available in the weather file.
    :type time_min_max: list
    :param lat_min_max: Cached min/max latitude values available in the weather file.
    :type lat_min_max: list
    :param lon_min_max: Cached min/max longitude values available in the weather file.
    :type lon_min_max: list
    """
    weather_path: str  # path to netCDF containing weather data

    def __init__(self, ship_config: ShipConfig):
        self.under_keel_clearance = ship_config.BOAT_UNDER_KEEL_CLEARANCE * u.meter
        self.draught_aft = ship_config.BOAT_DRAUGHT_AFT * u.meter
        self.draught_fore = ship_config.BOAT_DRAUGHT_FORE * u.meter

        self.time_min_max = [None, None]
        self.lat_min_max = [None, None]
        self.lon_min_max = [None, None]

    def get_required_water_depth(self):
        """Return required water depth in metres.

        The required water depth is computed as the maximum of fore and aft
        draught plus the under-keel clearance.

        :return: required water depth in metres (float)
        :rtype: float
        """
        needs_water_depth = max(self.draught_aft, self.draught_fore) + self.under_keel_clearance
        return needs_water_depth.value

    def get_ship_parameters(self, courses, lats, lons, time, speed, unique_coords=False):
        """Return `ShipParams` for the requested positions and times.

        Sub-classes must override this method to provide vessel-specific power,
        RPM and fuel estimations.

        :param courses: course angles for each routing segment (radians or degrees
            depending on caller conventions).
        :type courses: array-like
        :param lats: latitudes of start points for each routing segment.
        :type lats: array-like
        :param lons: longitudes of start points for each routing segment.
        :type lons: array-like
        :param time: start times for each routing segment (array of datetimes).
        :type time: array-like
        :param speed: speeds to evaluate at (one per segment).
        :type speed: array-like
        :param unique_coords: if True, the implementation may assume coordinates
            are unique and optimise lookups accordingly.
        :type unique_coords: bool
        :return: a `ShipParams` instance populated for each requested segment.
        :rtype: ShipParams
        """
        raise NotImplementedError()

    def print_init(self):
        """Log basic boat initialisation information.

        Implementations should use the project's logging/formatting helpers to
        print relevant initialisation values (e.g. fuel rate or geometry).
        """
        return None

    def evaluate_weather(self, ship_params, lats, lons, time):
        """Populate weather-related fields of a `ShipParams` object.

        This method reads the NetCDF file referenced by ``self.weather_path`` and
        interpolates (nearest) the required variables to the provided
        coordinates and times. The populated fields on ``ship_params`` include
        wave height, wave period, wave direction, wind and current components,
        pressure, temperatures and salinity.

        :param ship_params: `ShipParams` instance to populate.
        :type ship_params: ShipParams
        :param lats: latitudes for the lookups.
        :type lats: array-like
        :param lons: longitudes for the lookups.
        :type lons: array-like
        :param time: times for the lookups (array-like, datetime-like objects).
        :type time: array-like
        :return: the same ``ship_params`` instance populated with weather fields.
        :rtype: ShipParams
        """
        weather_data = xr.open_dataset(self.weather_path)
        n_coords = len(lats)

        wave_height = []
        wave_direction = []
        wave_period = []
        u_wind_speed = []
        v_wind_speed = []
        u_currents = []
        v_currents = []
        pressure = []
        air_temperature = []
        salinity = []
        water_temperature = []

        if self.time_min_max == [None, None]:
            self.time_min_max = [weather_data['time'].min(), weather_data['time'].max()]
            self.lat_min_max = [weather_data['latitude'].min(), weather_data['latitude'].max()]
            self.lon_min_max = [weather_data['longitude'].min(), weather_data['longitude'].max()]

            # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            # print('Coverage of weather data: ')
            # print(f'time range: {self.time_min_max[0]} - {self.time_min_max[1]}')
            # print(f'latitude range: {self.lat_min_max[0]} - {self.lat_min_max[1]}')
            # print(f'longitude range: {self.lon_min_max[0]} - {self.lon_min_max[1]}')
            # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

        for i_coord in range(0, n_coords):
            wave_direction.append(
                self.approx_weather(weather_data['VMDR'], lats[i_coord], lons[i_coord], time[i_coord]))
            wave_period.append(self.approx_weather(weather_data['VTPK'], lats[i_coord], lons[i_coord], time[i_coord]))
            wave_height.append(self.approx_weather(weather_data['VHM0'], lats[i_coord], lons[i_coord], time[i_coord]))
            v_currents.append(
                self.approx_weather(weather_data['vtotal'], lats[i_coord], lons[i_coord], time[i_coord], None, 0.5))
            u_currents.append(
                self.approx_weather(weather_data['utotal'], lats[i_coord], lons[i_coord], time[i_coord], None, 0.5))
            pressure.append(
                self.approx_weather(weather_data['Pressure_reduced_to_MSL_msl'], lats[i_coord], lons[i_coord],
                                    time[i_coord]))
            water_temperature.append(
                self.approx_weather(weather_data['thetao'], lats[i_coord], lons[i_coord], time[i_coord], None, 0.5))
            salinity.append(
                self.approx_weather(weather_data['so'], lats[i_coord], lons[i_coord], time[i_coord], None, 0.5))
            air_temperature.append(
                self.approx_weather(weather_data['Temperature_surface'], lats[i_coord], lons[i_coord], time[i_coord]))
            u_wind_speed.append(
                self.approx_weather(weather_data['u-component_of_wind_height_above_ground'], lats[i_coord],
                                    lons[i_coord], time[i_coord], 10))
            v_wind_speed.append(
                self.approx_weather(weather_data['v-component_of_wind_height_above_ground'], lats[i_coord],
                                    lons[i_coord], time[i_coord], 10))

        ship_params.wave_direction = np.array(wave_direction, dtype='float32') * u.radian
        ship_params.wave_period = np.array(wave_period, dtype='float32') * u.second
        ship_params.wave_height = np.array(wave_height, dtype='float32') * u.meter
        ship_params.u_wind_speed = np.array(u_wind_speed, dtype='float32') * u.meter / u.second
        ship_params.v_wind_speed = np.array(v_wind_speed, dtype='float32') * u.meter / u.second
        ship_params.v_currents = np.array(v_currents, dtype='float32') * u.meter / u.second
        ship_params.u_currents = np.array(u_currents, dtype='float32') * u.meter / u.second
        ship_params.pressure = np.array(pressure, dtype='float32') * u.kg / (u.meter * u.second ** 2)
        ship_params.air_temperature = np.array(air_temperature, dtype='float32') * u.Kelvin
        ship_params.air_temperature = ship_params.air_temperature.to(u.deg_C, equivalencies=u.temperature())
        ship_params.salinity = np.array(salinity, dtype='float32') * 0.001 * u.dimensionless_unscaled
        ship_params.water_temperature = np.array(water_temperature, dtype='float32') * u.deg_C

        weather_data.close()

        return ship_params

    def check_value_in_range(self, lats, lons, time):
        """Raise `ValueError` if any requested coordinate or time is outside
        the cached weather data ranges.

        The method uses the cached ``lat_min_max``, ``lon_min_max`` and
        ``time_min_max`` values populated when ``evaluate_weather`` was first
        called. If a value is out of range, the corresponding available range is
        printed and a ``ValueError`` is raised.

        :param lats: latitudes to check.
        :type lats: array-like
        :param lons: longitudes to check.
        :type lons: array-like
        :param time: times to check.
        :type time: array-like
        :raises ValueError: if any coordinate/time lies outside the available data.
        """
        if (lats > self.lat_min_max[1] or lats < self.lat_min_max[0]).any():
            weather_data = xr.open_dataset(self.weather_path)
            print(f'lat: {weather_data["latitude"].min().to_numpy()} - {weather_data["latitude"].max().to_numpy()}')
            raise ValueError(f'Latitude {lats} is out of weather range.')
        if (lons > self.lon_min_max[1] or lons < self.lon_min_max[0]).any():
            weather_data = xr.open_dataset(self.weather_path)
            print(f'lon: {weather_data["longitude"].min().to_numpy()} - {weather_data["longitude"].max().to_numpy()}')
            raise ValueError(f'Longitude {lons} is out of weather range.')
        if (np.datetime64(time) > self.time_min_max[1] or np.datetime64(time) < self.time_min_max[0]).any():
            weather_data = xr.open_dataset(self.weather_path)
            print(f'time: {weather_data["time"].min()} - {weather_data["time"].max()}')
            raise ValueError(f'Time {time} is out of weather range.')

    def approx_weather(self, var, lats, lons, time, height=None, depth=None):
        """Select nearest values from an xarray Variable and return as NumPy.

        Uses ``xarray.DataArray.sel`` with ``method='nearest'`` and fills
        missing values with zero. Optionally selects by ``height_above_ground``
        or ``depth`` where available.

        :param var: xarray variable (DataArray) to sample from.
        :type var: xarray.DataArray
        :param lats: latitude values for the lookup.
        :type lats: array-like
        :param lons: longitude values for the lookup.
        :type lons: array-like
        :param time: time values for the lookup.
        :type time: array-like
        :param height: optional height above ground to select (e.g. wind levels).
        :type height: float or None
        :param depth: optional depth to select (e.g. ocean fields).
        :type depth: float or None
        :return: sampled values as a NumPy array with NaNs replaced by 0.
        :rtype: numpy.ndarray
        """

        # self.check_value_in_range(lats, lons, time)

        ship_var = var.sel(latitude=lats, longitude=lons, time=time, method='nearest', drop=False)
        if height:
            ship_var = ship_var.sel(height_above_ground=height, method='nearest', drop=False)
        if depth:
            ship_var = ship_var.sel(depth=depth, method='nearest', drop=False)
        ship_var = ship_var.fillna(0).to_numpy()

        return ship_var

    def load_data(self):
        """Optional hook to (re)load vessel-specific data.

        Child classes may implement this to load auxiliary data (e.g. lookup
        tables) required for power/fuel computations.
        """
        return None

    def check_data_meaningful(self):
        """
        This is an optional method to check if default boat variables have been changed into meaningful values.
        It can be implemented in Child classes.
        """
        pass


class ConstantFuelBoat(Boat):
    fuel_rate: float  # dummy value for fuel_rate that is returned

    def __init__(self, ship_config: ShipConfig):
        super().__init__(ship_config)

        # mandatory variables
        self.fuel_rate = ship_config.BOAT_FUEL_RATE * u.kg / u.second

    def print_init(self):
        logger.info(form.get_log_step('boat fuel rate' + str(self.fuel_rate), 1))
        form.print_line()

    def get_ship_parameters(self, courses, lats, lons, time, speed, unique_coords=False):
        debug = False
        n_requests = len(courses)

        dummy_array = np.full(n_requests, -99)
        fuel_array = np.full(n_requests, self.fuel_rate)

        ship_params = ShipParams(
            fuel_rate=fuel_array * u.kg / u.s,
            power=dummy_array * u.Watt,
            rpm=dummy_array * u.Hz,
            speed=dummy_array * u.meter / u.second,
            r_wind=dummy_array * u.N,
            r_calm=dummy_array * u.N,
            r_waves=dummy_array * u.N,
            r_shallow=dummy_array * u.N,
            r_roughness=dummy_array * u.N,
            wave_height=dummy_array * u.meter,
            wave_direction=dummy_array * u.radian,
            wave_period=dummy_array * u.second,
            u_currents=dummy_array * u.meter / u.second,
            v_currents=dummy_array * u.meter / u.second,
            u_wind_speed=dummy_array * u.meter / u.second,
            v_wind_speed=dummy_array * u.meter / u.second,
            pressure=dummy_array * u.kg / u.meter / u.second ** 2,
            air_temperature=dummy_array * u.deg_C,
            salinity=dummy_array * u.dimensionless_unscaled,
            water_temperature=dummy_array * u.deg_C,
            status=dummy_array,
            message=np.full(n_requests, "")
        )

        if (debug):
            ship_params.print()
            form.print_step('fuel result' + str(ship_params.get_fuel_rate()))

        return ship_params
