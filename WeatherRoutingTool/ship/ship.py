import logging
from pathlib import Path

import numpy as np
import xarray as xr
from astropy import units as u

import WeatherRoutingTool.utils.formatting as form
from WeatherRoutingTool.ship.shipparams import ShipParams
from WeatherRoutingTool.ship.ship_config import ShipConfig
from WeatherRoutingTool.ship.evaluateSavedModels import SavedModelEvaluator

logger = logging.getLogger('WRT.ship')


# Boat: Main class for boats. Classes 'Tanker' and 'SailingBoat' derive from it
# Tanker: implements interface to mariPower package which is used for power estimation.

class Boat:
    speed: float  # boat speed in m/s
    weather_path: str  # path to netCDF containing weather data

    def __init__(self, init_mode='from_file', file_name=None, config_dict=None):
        config_obj = None
        if init_mode == "from_file":
            config_obj = ShipConfig.assign_config(Path(file_name))
        else:
            config_obj = ShipConfig.assign_config(init_mode='from_dict', config_dict=config_dict)

        self.speed = config_obj.BOAT_SPEED * u.meter / u.second
        self.under_keel_clearance = config_obj.BOAT_UNDER_KEEL_CLEARANCE * u.meter
        self.draught_aft = config_obj.BOAT_DRAUGHT_AFT * u.meter
        self.draught_fore = config_obj.BOAT_DRAUGHT_FORE * u.meter

    def get_required_water_depth(self):
        needs_water_depth = max(self.draught_aft, self.draught_fore) + self.under_keel_clearance
        return needs_water_depth.value

    def get_ship_parameters(self, courses, lats, lons, time, speed=None, unique_coords=False):
        pass

    def get_boat_speed(self):
        return self.speed

    def print_init(self):
        pass

    def set_boat_speed(self, speed):
        self.speed = speed

    def evaluate_weather(self, ship_params, lats, lons, time):
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

        return ship_params

    def approx_weather(self, var, lats, lons, time, height=None, depth=None):
        ship_var = var.sel(latitude=lats, longitude=lons, time=time, method='nearest', drop=False)
        if height:
            ship_var = ship_var.sel(height_above_ground=height, method='nearest', drop=False)
        if depth:
            ship_var = ship_var.sel(depth=depth, method='nearest', drop=False)
        ship_var = ship_var.fillna(0).to_numpy()

        return ship_var

    def load_data(self):
        pass

    def check_data_meaningful(self):
        """
        This is an optional method to check if default boat variables have been changed into meaningful values.
        It can be implemented in Child classes.
        """
        pass


class ConstantFuelBoat(Boat):
    fuel_rate: float  # dummy value for fuel_rate that is returned
    speed: float  # boat speed

    def __init__(self, init_mode='from_file', file_name=None, config_dict=None):
        super().__init__(init_mode, file_name, config_dict)
        config_obj = None
        if init_mode == "from_file":
            config_obj = ShipConfig.assign_config(Path(file_name))
        else:
            config_obj = ShipConfig.assign_config(init_mode='from_dict', config_dict=config_dict)

        # mandatory variables
        self.fuel_rate = config_obj.BOAT_FUEL_RATE * u.kg / u.second

    def print_init(self):
        logger.info(form.get_log_step('boat speed' + str(self.speed), 1))
        logger.info(form.get_log_step('boat fuel rate' + str(self.fuel_rate), 1))
        form.print_line()

    def get_ship_parameters(self, courses, lats, lons, time, speed=None, unique_coords=False):
        debug = False
        n_requests = len(courses)

        dummy_array = np.full(n_requests, -99)
        fuel_array = np.full(n_requests, self.fuel_rate)
        speed_array = np.full(n_requests, self.speed)

        ship_params = ShipParams(
            fuel_rate=fuel_array * u.kg / u.s,
            power=dummy_array * u.Watt,
            rpm=dummy_array * u.Hz,
            speed=speed_array * u.meter / u.second,
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


class NNBoat(Boat):
    model_path: str
    evaluator: SavedModelEvaluator
    depth_data: xr
    weather_path: str
    draught: float
    nominal_power: float
    P_perc: np.array
    feature_names: list

    def __init__(self, init_mode='from_file', file_name=None, config_dict=None):
        super().__init__(init_mode, file_name, config_dict)
        config_obj = None
        if init_mode == "from_file":
            config_obj = ShipConfig.assign_config(Path(file_name))
        else:
            config_obj = ShipConfig.assign_config(init_mode='from_dict', config_dict=config_dict)

        # mandatory variables
        self.model_path = config_obj.BOAT_NNMODEL_PATH
        self.draught = (config_obj.BOAT_DRAUGHT_AFT + config_obj.BOAT_DRAUGHT_FORE) / 2

        self.evaluator = SavedModelEvaluator()
        info = self.evaluator.get_model_info(self.model_path)

        print("\nNN Model Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        if not config_obj.DEPTH_DATA == " ":
            self.use_depth_data = True
            self.depth_data = xr.open_dataset(config_obj.DEPTH_DATA)
        self.weather_path = config_obj.WEATHER_DATA

        self.nominal_power = config_obj.BOAT_SMCR_POWER * u.kiloWatt
        self.nominal_power = self.nominal_power.to(u.Watt)
        self.fuel_rate = config_obj.BOAT_FUEL_RATE * u.gram / (u.kiloWatt * u.hour)
        self.fuel_rate = self.fuel_rate.to(u.kg / (u.Watt * u.second))
        self.P_perc = np.array([])

        self.feature_names = [
            'STW',  # speed through water (m/s)
            'draft_fp_interpolated_between_low_speeds',  # fore draft (m)
            'draft_ap_interpolated_between_low_speeds',  # after draft (m)
            'rel_wind_direction',  # relative wind direction (0-360°)
            'thetao',  # water temperature (°C)
            'Temperature_surface',  # air temperatrue (°K)
            'rel_seaway_direction',  # wave direction (0-360°)
            'z',  # water depth (m)
            'Pressure_reduced_to_MSL_msl',  # pressure (Pa)
            'VHM0',  # wave height (m)
            'VTPK',  # wave period
            'so',  # salinity
            'ucomponent_of_wind_height_above_ground',  # u component wind speed (m/s)
            'vcomponent_of_wind_height_above_ground'  # v component wind speed (m/s)
        ]

    def get_relative_wind_dir(self, ang_boat, ang_wind):
        """
            calculate relative wind direction [0°,180°] between ship course and true wind direction

            - head wind: 0°
            - tail wind: 180°
        """

        delta_ang = ang_wind - ang_boat

        delta_ang[delta_ang < 0 * u.degree] = abs(delta_ang[delta_ang < 0 * u.degree])
        delta_ang[delta_ang > 180 * u.degree] = abs(360 * u.degree - delta_ang[delta_ang > 180 * u.degree])

        return delta_ang

    def coordinate_transformation(self, input_dict):
        input_dict['Temperature_surface'] = input_dict['Temperature_surface'] + 274.15
        input_dict['so'] = input_dict['so'] * 1000
        return input_dict

    def get_input_data(self, input_dict):
        input_data = np.array([[]])
        for feature in input_dict:
            input_data = np.append(input_data,input_dict[feature])
        return input_data

    def get_ship_parameters(self, courses, lats, lons, time, speed=None, unique_coords=False):
        debug = False
        n_requests = len(courses)

        # initialise clean ship params object
        dummy_array = np.full(n_requests, -99)
        speed_array = np.full(n_requests, self.speed)

        ship_params = ShipParams(
            fuel_rate=dummy_array * u.kg / u.s,
            power=dummy_array * u.Watt,
            rpm=dummy_array * u.Hz,
            speed=speed_array * u.meter / u.second,
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
        # calculate added resistances & update ShipParams object respectively; update also for environmental conditions
        ship_params = self.evaluate_weather(ship_params, lats, lons, time)

        absolute_wind_direction = 180 + 180 / np.pi * np.arctan2(ship_params.u_wind_speed.value,
                                                                 ship_params.v_wind_speed.value)
        absolute_wind_direction = (absolute_wind_direction % 360) * u.degree
        rel_wind_direction = self.get_relative_wind_dir(courses, absolute_wind_direction)

        absolute_seaway_direction = 180 + 180 / np.pi * np.arctan2(ship_params.u_currents.value,
                                                                   ship_params.v_currents.value)
        absolute_seaway_direction = (absolute_seaway_direction % 360) * u.degree
        rel_seaway_direction = self.get_relative_wind_dir(courses, absolute_seaway_direction)

        lat_da = xr.DataArray(lats, dims="dummy")
        lon_da = xr.DataArray(lons, dims="dummy")
        rounded_ds = self.depth_data["z"].interp(latitude=lat_da, longitude=lon_da, method="linear")
        depth = rounded_ds.to_numpy()

        array_shape = ship_params.water_temperature.shape
        speed = np.full(array_shape[0], self.speed) * 1.994
        draught = np.full(array_shape[0], self.draught)
        P_perc = np.full(array_shape[0], -99)

        print('array_shape: ', array_shape[0])
        print('speed: ', type(speed[0]))
        print('draugth:', type(draught[0]))
        print('water_temp: ', type(ship_params.water_temperature[0].value))

        for ipoint in range(len(lats)):
            input_dict = {
                'STW': speed[ipoint],  # STW
                'draft_fp_interpolated_between_low_speeds': draught[ipoint],  # draft_fp_interpolated_between_low_speeds
                'draft_ap_interpolated_between_low_speeds': draught[ipoint],  # draft_ap_interpolated_between_low_speeds
                'rel_wind_direction': rel_wind_direction[ipoint].value,  # rel_wind_direction
                'thetao': ship_params.water_temperature[ipoint].value,  # thetao
                'Temperature_surface': ship_params.air_temperature[ipoint].value,  # Temperature_surface
                'rel_seaway_direction': rel_seaway_direction[ipoint].value,  # rel_seaway_direction
                'z': depth[ipoint],  # z
                'Pressure_reduced_to_MSL_msl': ship_params.pressure[ipoint].value,  # Pressure_reduced_to_MSL_msl
                'VHM0': ship_params.wave_height[ipoint].value,  # VHM0
                'VTPK': ship_params.wave_period[ipoint].value,  # VTPK
                'so': ship_params.salinity[ipoint].value,  # so
                'ucomponent_of_wind_height_above_ground': ship_params.u_wind_speed[ipoint].value,  # u_wind
                'vcomponent_of_wind_height_above_ground': ship_params.v_wind_speed[ipoint].value  # v_wind
            }
            print('before conversion: ', input_dict)
            input_dict = self.coordinate_transformation(input_dict)
            print('after conversion: ', input_dict)
            input_data= self.get_input_data(input_dict)
            print('input_data: ', input_data)
            P_perc[ipoint] = self.evaluator.evaluate(model_path=self.model_path, input_data=input_data)
            print('prediction: ', P_perc[ipoint])

        self.P_perc = np.append(self.P_perc, P_perc)
        prediction = P_perc / 100 * self.nominal_power

        ship_params.power = prediction
        ship_params.fuel_rate = self.fuel_rate * prediction

        ship_params.print()
        return ship_params
