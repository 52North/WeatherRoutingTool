import logging
import os
import sys
from pathlib import Path

import dill
import numpy as np
import torch
import xarray as xr
from astropy import units as u

from surrogate_lib.evaluation.test import ModelManager

import WeatherRoutingTool.utils.formatting as form
from WeatherRoutingTool.ship.shipparams import ShipParams
from WeatherRoutingTool.ship.ship import Boat
from WeatherRoutingTool.ship.ship_config import ShipConfig
from WeatherRoutingTool.ship.evaluateSavedModels import SavedModelEvaluator
from WeatherRoutingTool.weather import WeatherCond


class NNBoat(Boat):
    model_path: str
    evaluator: ModelManager
    depth_data: xr
    weather_path: str
    draught: float
    nominal_power: float
    P_perc: np.array
    feature_names: list

    def __init__(self, ship_config: ShipConfig):
        super().__init__(ship_config)

        # mandatory variables
        self.model_path = ship_config.BOAT_NNMODEL_PATH
        self.draught = (ship_config.BOAT_DRAUGHT_AFT + ship_config.BOAT_DRAUGHT_FORE) / 2

        # init nnmodel
        self.evaluator = ModelManager()

        depth_path = str(ship_config.DEPTH_DATA)
        if not depth_path == " ":
            self.use_depth_data = True
            self.depth_data = xr.open_dataset(ship_config.DEPTH_DATA)
        self.weather_path = ship_config.WEATHER_DATA

        self.nominal_power = ship_config.BOAT_SMCR_POWER * u.kiloWatt
        self.nominal_power = self.nominal_power.to(u.Watt)
        self.fuel_rate = ship_config.BOAT_FUEL_RATE * u.gram / (u.kiloWatt * u.hour)
        self.fuel_rate = self.fuel_rate.to(u.kg / (u.Watt * u.second))
        self.P_perc = np.array([])

        self.feature_names = [
            'STW',  # speed through water (m/s)
            'AP (interpolated)',  # after draft (m)
            'FP (interpolated)',  # fore draft (m)
            'WIND_SPEED_REL',  # wind speed
            'WIND_DIRECTION_REL',  # relative wind direction in deg, remapped to -180 to +180  (e.g. raw 210° → -150°)
            'rel_seaway_direction',  # relative wave direction in deg, remapped to -180 to +180
            'VHM0',  # wave height (m)
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

    def get_relative_wind_dir_asymmetric(self, ang_boat, ang_wind):
        """
            calculate relative wind direction [-180°, 180°] between ship course and true wind direction
        """

        delta_ang = ang_wind - ang_boat

        delta_ang = delta_ang % (360 * u.degree)
        print('delta_ang', delta_ang)

        delta_ang[delta_ang < -180 * u.degree] = delta_ang[delta_ang < -180 * u.degree] + 360 * u.degree
        delta_ang[delta_ang > 180 * u.degree] = delta_ang[delta_ang > 180 * u.degree] - 360 * u.degree
        print('delta_ang: ', delta_ang)

        return delta_ang

    def get_input_data(self, input_dict):
        input_data = np.array([[]])
        for feature in input_dict:
            input_data = np.append(input_data, input_dict[feature])
        return input_data

    def get_fuel_rate_from_power(self, n, P):
        filepath = ("/home/kdemmich/3_Software/MariDataEpilog_surrogate_lib/venv_martrans/lib/python3.11/"
                    "site-packages/mariPower/")
        n = np.full(P.shape, n)
        power = P.value / 1000

        fuel_model = dill.load(
            open(
                os.path.join(filepath, "data", "CBT_FOC_of_n_Power_quadratic_FDS.pcl"), "rb"
            )
        )

        fuel_rate = fuel_model.predict_values(
            np.vstack((n / 60, power / 1000)).T).squeeze()  # fuelConsumptionCBT.FuelConsumptionCBT(n,P)
        return fuel_rate / 1000 * u.kg / u.second

    def get_apparent_wind(self, speed, true_wind_speed, true_wind_angle):
        """
            calculate apparent wind speed from true wind and ship course
        """
        apparent_wind_speed = (speed * speed + true_wind_speed * true_wind_speed
                               + 2.0 * speed * true_wind_speed * np.cos(np.radians(true_wind_angle)))
        apparent_wind_speed = np.sqrt(apparent_wind_speed)

        angle_rad = np.radians(true_wind_angle.value)
        apparent_wind_angle = np.full(true_wind_angle.shape, - 99) * u.radian

        for iang in range(0, true_wind_speed.shape[0]):
            arg_arcsin = true_wind_speed[iang] * np.sin(np.radians(true_wind_angle[iang])) / apparent_wind_speed[
                iang] * u.radian

            # catch it if argument of arcsin is > 1 due to rounding issues but make sure to apply this only for
            # rounding issues

            abs_arcsin = np.abs(arg_arcsin)
            diff_to_one = abs_arcsin - 1 * u.radian
            if diff_to_one > 0:
                assert diff_to_one < 0.000001 * u.radian
                if arg_arcsin > 0:
                    arg_arcsin = 1 * u.radian
                else:
                    arg_arcsin = -1 * u.radian

            if apparent_wind_speed[iang] > 0:
                apparent_wind_angle[iang] = np.arcsin(arg_arcsin.value) * u.radian
            else:
                apparent_wind_angle[iang] = 0 * u.radian

            # catch it if psi > 90° as arcsin is only defined for 0 < psi < 90°
            # - calculate true wind angle 'true_ang_perp' for which apparent wind angle is 90°
            # - if true wind angle is larger than 'true_ang_perp', subtract pi from apparent wind angle
            # - apparent wind angle is always < 90° if boat speed > true wind speed; skip correction here
            arg_arccos = speed[iang] / true_wind_speed[iang]
            if arg_arccos > 1:
                continue
            true_ang_perp = np.pi * u.radian - np.arccos(speed[iang] / true_wind_speed[iang])
            if angle_rad[iang] * u.radian > true_ang_perp:
                apparent_wind_angle[iang] = np.pi * u.radian - apparent_wind_angle[iang]
            if -angle_rad[iang] * u.radian > true_ang_perp:
                apparent_wind_angle[iang] = -np.pi * u.radian - apparent_wind_angle[iang]

            if np.isnan(apparent_wind_angle[iang]):
                print('true_wind_speed: ', true_wind_speed[iang])
                print('apparent_wind_speed: ', apparent_wind_speed[iang])
                print('true_wind_angle: ', true_wind_angle[iang])
                print('true_ang_perp: ', true_ang_perp)
                print('angle_rad: ', angle_rad[iang])
                print('arg_arcsin: ', arg_arcsin)
                print('arg_arccos: ', arg_arccos)
                raise ValueError('Apparent wind angle is nan!')

        apparent_wind_angle = apparent_wind_angle.to(u.degree)
        return {'app_wind_speed': apparent_wind_speed, 'app_wind_angle': apparent_wind_angle}

    def load_normalization(self, model_path: str):
        """Extract x_mean and x_std stored inside the .pth checkpoint."""
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        params = checkpoint.get("normalization_params", {})
        x_mean = params.get("x_mean")
        x_std = params.get("x_std")
        if x_mean is None or x_std is None:
            raise ValueError(
                f"x_mean/x_std not found in {model_path}. "
                "Only final_model.pth files are supported, not fold models."
            )
        if isinstance(x_mean, torch.Tensor):
            x_mean = x_mean.cpu().numpy()
        if isinstance(x_std, torch.Tensor):
            x_std = x_std.cpu().numpy()
        return x_mean, x_std

    def preprocess(self, X_raw: np.ndarray, model_type: str) -> np.ndarray:
        """
        Convert raw user input (7 features, angles in degrees) to the 9-feature
        sin/cos encoded format expected by the models.

        Parameters
        ----------
        X_raw : np.ndarray, shape (n_samples, 7)
            Columns in INPUT_FEATURES order:
            STW, AP, FP, WIND_SPEED_REL, WIND_DIRECTION_REL, VHM0, rel_seaway_direction

        Returns
        -------
        np.ndarray, shape (n_samples, 9)
            Columns in MODEL_FEATURES order.
        """
        X_raw = np.atleast_2d(X_raw).astype(float)

        if model_type == "smallDataset_boardWind_9features" or model_type == "NNfinal":
            stw = X_raw[:, 0]
            ap = X_raw[:, 1]
            fp = X_raw[:, 2]
            wind_speed = X_raw[:, 3]
            wind_dir_deg = self._wrap(X_raw[:, 4])
            vhm0 = X_raw[:, 5]
            sea_dir_deg = self._wrap(X_raw[:, 6])

            wind_rad = np.deg2rad(wind_dir_deg)
            sea_rad = np.deg2rad(sea_dir_deg)

            return np.column_stack([
                stw,
                ap,
                fp,
                wind_speed,
                np.sin(wind_rad),
                np.cos(wind_rad),
                vhm0,
                np.sin(sea_rad),
                np.cos(sea_rad),
            ])
        if model_type == "smallDataset_Copernicus":
            stw = X_raw[:, 0]
            ap = X_raw[:, 1]
            fp = X_raw[:, 2]
            wind_speed = X_raw[:, 3]
            wind_dir_deg = self._wrap(X_raw[:, 4])
            vhm0 = X_raw[:, 5]
            sea_dir_deg = self._wrap(X_raw[:, 6])

            wind_rad = np.deg2rad(wind_dir_deg)
            sea_rad = np.deg2rad(sea_dir_deg)

            return np.column_stack([
                stw,
                ap,
                fp,
                wind_speed,
                wind_dir_deg,
                vhm0,
                sea_dir_deg,
            ])
        if model_type == "STWAndWaves":
            stw = X_raw[:, 0]
            vhm0 = X_raw[:, 1]
            sea_dir_deg = self._wrap(X_raw[:, 2])
            sea_rad = np.deg2rad(sea_dir_deg)

            return np.column_stack([
                stw,
                vhm0,
                np.sin(sea_rad),
                np.cos(sea_rad),
            ])

        if model_type == "STWAndWind":
            stw = X_raw[:, 0]
            wind_speed = X_raw[:, 1]
            wind_dir_deg = self._wrap(X_raw[:, 2])
            wind_rad = np.deg2rad(wind_dir_deg)

            return np.column_stack([
                stw,
                wind_speed,
                np.sin(wind_rad),
                np.cos(wind_rad),
            ])
        raise NotImplementedError(f"The model_type {model_type} not implemented.")

    def predict_mean(self, manager: ModelManager, model_path: str,
                     X_raw: np.ndarray, model_type: str, device: str = "cpu") -> np.ndarray:
        """
        Predict in original (denormalized) target units.

        Parameters
        ----------
        X_raw : np.ndarray, shape (n_samples, 3)
            Raw input features in INPUT_FEATURES order. Angle in degrees.

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        X = self.preprocess(X_raw, model_type)
        x_mean, x_std = self.load_normalization(model_path)
        return manager.predict(model_path, X, x_mean=x_mean, x_std=x_std, device=device).flatten()

    def _wrap(self, deg: np.ndarray) -> np.ndarray:
        """Wrap any degree value(s) to the ±180° range."""
        return (np.asarray(deg, dtype=float) + 180.0) % 360.0 - 180.0

    def predict_with_uncertainty(self, manager: ModelManager, model_path: str,
                                 X_raw: np.ndarray, model_type: str, device: str = "cpu"):
        """
        Predict mean and 1-sigma std for GP models (denormalized).

        Parameters
        ----------
        X_raw : np.ndarray, shape (n_samples, 7)
            Raw input features in INPUT_FEATURES order. Angles in degrees.

        Returns
        -------
        mean : np.ndarray, shape (n_samples,)
        std  : np.ndarray, shape (n_samples,)
        """
        X = self.preprocess(X_raw, model_type)
        x_mean, x_std = self.load_normalization(model_path)
        mean, std = manager.predict_with_uncertainty(
            model_path, X, x_mean=x_mean, x_std=x_std, device=device
        )
        return mean.flatten(), std.flatten()

    def get_ship_parameters(self, courses, lats, lons, time, speed, unique_coords=False):
        debug = False
        n_requests = len(courses)

        # initialise clean ship params object
        dummy_array = np.full(n_requests, -99)

        ship_params = ShipParams(
            fuel_rate=dummy_array * u.kg / u.s,
            power=dummy_array * u.Watt,
            rpm=dummy_array * u.Hz,
            speed=speed,
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

        # calculate true wind speed and direction and project to 0°-180°
        absolute_wind_direction = WeatherCond.get_theta_from_uv(ship_params.u_wind_speed.value,
                                                                ship_params.v_wind_speed.value)
        absolute_wind_direction = (absolute_wind_direction % 360) * u.degree
        absolute_wind_speed = (np.sqrt((ship_params.u_wind_speed.value * ship_params.u_wind_speed.value
                                        + ship_params.v_wind_speed.value * ship_params.v_wind_speed.value))
                               * u.meter / u.second)

        # calculate apparent wind speed and wind direction in boat coordinate system
        relative_wind_direction = self.get_relative_wind_dir_asymmetric(courses, absolute_wind_direction)
        wind_res = self.get_apparent_wind(speed, absolute_wind_speed, relative_wind_direction)

        if debug:
            print('courses: ', courses)
            print('absolute wind direction: ', absolute_wind_direction)
            print('relative wind direction: ', relative_wind_direction)
            print('relative wind direction converted:', wind_res['app_wind_angle'])
            print('true wind speed: ', absolute_wind_speed)
            print('apparent wind speed', wind_res['app_wind_speed'])
            print('u: ', ship_params.u_wind_speed.value)
            print('v: ', ship_params.v_wind_speed.value)

        absolute_seaway_direction = ship_params.wave_direction
        rel_seaway_direction = self.get_relative_wind_dir_asymmetric(courses, absolute_seaway_direction)

        # lat_da = xr.DataArray(lats, dims="dummy")
        # lon_da = xr.DataArray(lons, dims="dummy")
        # rounded_ds = self.depth_data["z"].interp(latitude=lat_da, longitude=lon_da, method="linear")
        # depth = rounded_ds.to_numpy()

        array_shape = ship_params.water_temperature.shape
        speed = np.full(array_shape[0], speed)
        draught = np.full(array_shape[0], self.draught)
        P_perc = np.full(array_shape[0], -99)

        if debug:
            print('array_shape: ', array_shape[0])
            print('speed: ', type(speed[0]))
            print('draugth:', type(draught[0]))
            print('water_temp: ', type(ship_params.water_temperature[0].value))

        for ipoint in range(len(lats)):
            input_dict = {
                'STW': speed[ipoint],  # STW
                'AP (interpolated)': draught[ipoint],
                'FP (interpolated)': draught[ipoint],
                'WIND_SPEED_REL': wind_res['app_wind_speed'][ipoint].value,
                'WIND_DIRECTION_REL': wind_res['app_wind_angle'][ipoint].value,
                'VHM0': ship_params.wave_height[ipoint].value,  # VHM0
                'rel_seaway_direction': rel_seaway_direction[ipoint].value,  # rel_seaway_direction
            }
            if debug:
                print('input_dict: ', input_dict)
            input_data = self.get_input_data(input_dict)
            if debug:
                print('input_data: ', input_data)
            model_type = "NNfinal"
            pred = self.predict_mean(self.evaluator, self.model_path, input_data, model_type)
            P_perc[ipoint] = pred[0]
            if debug:
                print('prediction: ', P_perc[ipoint])

        self.P_perc = np.append(self.P_perc, P_perc)
        prediction = P_perc / 100 * self.nominal_power

        ship_params.power = prediction
        ship_params.fuel_rate = self.fuel_rate * prediction

        return ship_params
