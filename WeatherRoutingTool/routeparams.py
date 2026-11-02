"""
Provide the container class ``RouteParams`` for a route and its ship parameters.

The module contains functions to write routes to and read them from GeoJSON files, to read measured routes from
parquet and csv files and to plot route properties like power, fuel consumption and speed.
"""
import json
from datetime import datetime, timedelta

import cartopy.crs as ccrs
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas
from geovectorslib import geod
from matplotlib import gridspec
from astropy import units as u

import WeatherRoutingTool.utils.unit_conversion as units
import WeatherRoutingTool.utils as utils
import WeatherRoutingTool.utils.graphics as graphics
import WeatherRoutingTool.utils.formatting as form
from WeatherRoutingTool.utils.formatting import NumpyArrayEncoder
from WeatherRoutingTool.ship.shipparams import ShipParams

logger = logging.getLogger('WRT.Routeparams')


class RouteParams:
    """
    This is a Container class for route parameters. Parameters are assumed to be constant for every route segment (=
    distance from one coordinate pair to the following). The parameters for each routing step are tied to the start
    point of each route segment. Since the number of routing steps is by one smaller than the number of coordinate
    pairs, the arrays of this container class exhibit different lengths:
        - Type 1: count + 1: dists_per_step, ship_params_per_step, courses_per_step
        - Type 2: count + 2: lats_per_step, lons_per_step, starttime_per_step
    When the RouteParams object is written to json file, the full arrays of type 2 are written. The corresponding values
    from the arrays of type 1 are set to -99 for the last coordinate pair.

    :param count: number of routing steps minus one, i.e. the number of route segments is ``count + 1``
    :type count: int
    :param start: latitude and longitude of the start point in degrees
    :type start: tuple
    :param finish: latitude and longitude of the destination in degrees
    :type finish: tuple
    :param gcr: distance travelled from start to destination in m; None if not determined
    :type gcr: astropy.units.Quantity or None
    :param route_type: name of the route, e.g. ``'min_fuel_route'``
    :type route_type: str
    :param time: travel time from start to destination
    :type time: astropy.units.Quantity or datetime.timedelta
    :param lats_per_step: latitude at the beginning of each step and of the destination in degrees (type 2)
    :type lats_per_step: numpy.ndarray
    :param lons_per_step: longitude at the beginning of each step and of the destination in degrees (type 2)
    :type lons_per_step: numpy.ndarray
    :param course_per_step: course of each step in degrees (type 1)
    :type course_per_step: astropy.units.Quantity
    :param dists_per_step: distance travelled on the great circle for every step in m (type 1)
    :type dists_per_step: astropy.units.Quantity
    :param starttime_per_step: time at the beginning of each step and arrival time at the destination (type 2)
    :type starttime_per_step: numpy.ndarray
    :param ship_params_per_step: ship parameters for every step (type 1)
    :type ship_params_per_step: ShipParams
    """

    count: int  # routing step (starting from 0)
    start: tuple  # lat, lon at start point (0 - 360°)
    finish: tuple  # lat, lon of destination (0 - 360°)
    gcr: tuple  # distance from start to end on great circle (0 - 360°)
    route_type: str  # route name
    time: timedelta  # travel time needed for the route (s)

    ship_params_per_step: ShipParams  # ship parameters per routing step
    lats_per_step: tuple  # latitude at beginning of each step + latitude destination (0-360°)
    lons_per_step: tuple  # longitude at beginning of each step + longitude destination (0-360°)
    course_per_step: tuple  # course per step (0-360°)
    dists_per_step: tuple  # distance traveled on great circle for every step (m)
    starttime_per_step: tuple  # start time at beginning of each step + time when destination is reached

    def __init__(self, count, start, finish, gcr, route_type, time, lats_per_step, lons_per_step, course_per_step,
                 dists_per_step, starttime_per_step, ship_params_per_step):
        self.count = count
        self.start = start
        self.finish = finish
        self.gcr = gcr
        self.route_type = route_type
        self.time = time
        self.lats_per_step = lats_per_step
        self.lons_per_step = lons_per_step
        self.course_per_step = course_per_step
        self.dists_per_step = dists_per_step
        self.starttime_per_step = starttime_per_step
        self.ship_params_per_step = ship_params_per_step

    def print_route(self):
        form.print_line()
        logger.info('Printing route:  ' + str(self.route_type))
        logger.info('Going from ' + str(self.start))
        logger.info('to ' + str(self.finish))
        logger.info('number of routing steps: ' + str(self.count))
        logger.info('latitude at start of each step: ' + str(self.lats_per_step))
        logger.info('longitude at start of each step: ' + str(self.lons_per_step))
        logger.info('distance per step: ' + str(self.dists_per_step))
        logger.info('courses for each step: ' + str(self.course_per_step))
        logger.info('gcr traveled per step (m): ' + str(self.dists_per_step))
        logger.info('time at start of each step: ' + str(self.starttime_per_step))

        self.ship_params_per_step.print()

        logger.info('full fuel consumed (kg): ' + str(self.get_full_fuel()))
        logger.info('full travel time (h): ' + str(self.time))
        logger.info('travel distance on great circle (m): ' + str(self.gcr))

        form.print_line()

    def __eq__(self, route2):
        bool_equal = True
        if not (self.count == route2.count):
            raise ValueError('Route counts not matching')
        if not (np.array_equal(self.start, route2.start)):
            raise ValueError('Route start not matching')
        if not (np.array_equal(self.finish, route2.finish)):
            raise ValueError('Route finsh not matching')
        if not (np.array_equal(self.time, route2.time)):
            raise ValueError('Route time not matching: self=' + str(self.time) + ' other=' + str(route2.time))
        if not (np.array_equal(self.ship_params_per_step.fuel_rate, route2.ship_params_per_step.fuel_rate)):
            raise ValueError('Route fuel not matching: self=' + str(self.ship_params_per_step.fuel_rate) +
                             ' other=' + str(route2.ship_params_per_step.fuel_rate))
        if not (np.array_equal(self.ship_params_per_step.rpm, route2.ship_params_per_step.rpm)):
            raise ValueError('Route rpm not matching')
        if not (np.array_equal(self.lats_per_step, route2.lats_per_step)):
            raise ValueError('Route lats_per_step not matching')
        if not (np.array_equal(self.lons_per_step, route2.lons_per_step)):
            raise ValueError('Route lons_per_step not matching')
        if not (np.array_equal(self.course_per_step, route2.course_per_step)):
            raise ValueError('Route course_per_step not matching')
        if not (np.array_equal(self.dists_per_step, route2.dists_per_step)):
            raise ValueError('Route dists_per_step not matching')

        return bool_equal

    def convert_to_dict(self):
        rp_dict = {"count": self.count, "start": self.start, "finish": self.finish, "route type": self.route_type,
                   "gcr": self.gcr, "time": self.time, "lats_per_step": self.lats_per_step,
                   "lons_per_step": self.lons_per_step, "course_per_step": self.course_per_step,
                   "dists_per_step": self.dists_per_step, "starttime_per_step": self.starttime_per_step}
        return rp_dict

    def write_to_geojson(self, filename):
        """
        Write the route as GeoJSON FeatureCollection of points to a file.

        Every coordinate pair (``count + 2`` points) is written as point feature with the coordinates in the order
        (longitude, latitude). For the destination, all ship parameters are set to -99.

        :param filename: path of the output file
        :type filename: str
        """
        rp_dict = {}
        rp_dict['type'] = 'FeatureCollection'
        feature_list = []

        logger.info(f'Write route parameters to {filename}')

        for i in range(0, self.count + 2):
            feature = {}
            geometry = {}
            properties = {}

            geometry['type'] = 'Point'
            # geometry['coordinates'] = [self.lats_per_step[i], self.lons_per_step[i]]
            geometry['coordinates'] = [self.lons_per_step[i], self.lats_per_step[i]]

            properties['time'] = self.starttime_per_step[i]
            if i == (self.count + 1):
                properties['speed'] = {'value': -99, 'unit': 'm/s'}
                properties['engine_power'] = {'value': -99, 'unit': 'kW'}
                properties['fuel_consumption'] = {'value': -99, 'unit': 't/h'}
                properties['fuel_type'] = self.ship_params_per_step.fuel_type
                properties['propeller_revolution'] = {'value': -99, 'unit': 'rpm'}
                properties['calm_resistance'] = {'value': -99, 'unit': 'N'}
                properties['wind_resistance'] = {'value': -99, 'unit': 'N'}
                properties['wave_resistance'] = {'value': -99, 'unit': 'N'}
                properties['shallow_water_resistance'] = {'value': -99, 'unit': 'N'}
                properties['hull_roughness_resistance'] = {'value': -99, 'unit': 'N'}
                properties['wave_height'] = {'value': -99, 'unit': 'm'}
                properties['wave_direction'] = {'value': -99, 'unit': 'radian'}
                properties['wave_period'] = {'value': -99, 'unit': 's'}
                properties['u_currents'] = {'value': -99, 'unit': 'm/s'}
                properties['v_currents'] = {'value': -99, 'unit': 'm/s'}
                properties['u_wind_speed'] = {'value': -99, 'unit': 'm/s'}
                properties['v_wind_speed'] = {'value': -99, 'unit': 'm/s'}
                properties['pressure'] = {'value': -99, 'unit': 'Pa'}
                properties['air_temperature'] = {'value': -99, 'unit': '°C'}
                properties['salinity'] = {'value': -99, 'unit': '-'}
                properties['water_temperature'] = {'value': -99, 'unit': '°C'}
                properties['status'] = {'value': -99}
                properties['message'] = {'value': ""}
            else:
                properties['speed'] = {'value': self.ship_params_per_step.speed[i].value, 'unit': 'm/s'}
                properties['engine_power'] = {'value': self.ship_params_per_step.power[i].to("kW").value, 'unit': 'kW'}
                properties['fuel_consumption'] = {
                    'value': self.ship_params_per_step.fuel_rate[i].to(u.tonne / u.hour).value,
                    'unit': 't/h'
                }
                properties['fuel_type'] = self.ship_params_per_step.fuel_type
                properties['propeller_revolution'] = {'value': self.ship_params_per_step.rpm[i].value, 'unit': 'rpm'}
                properties['calm_resistance'] = {'value': self.ship_params_per_step.r_calm[i].value, 'unit': 'N'}
                properties['wind_resistance'] = {'value': self.ship_params_per_step.r_wind[i].value, 'unit': 'N'}
                properties['wave_resistance'] = {'value': self.ship_params_per_step.r_waves[i].value, 'unit': 'N'}
                properties['shallow_water_resistance'] = {'value': self.ship_params_per_step.r_shallow[i].value,
                                                          'unit': 'N'}
                properties['hull_roughness_resistance'] = {'value': self.ship_params_per_step.r_roughness[i].value,
                                                           'unit': 'N'}
                properties['wave_height'] = {'value': self.ship_params_per_step.wave_height[i].value, 'unit': 'm'}
                properties['wave_direction'] = {'value': self.ship_params_per_step.wave_direction[i].value,
                                                'unit': 'radian'}
                properties['wave_period'] = {'value': self.ship_params_per_step.wave_period[i].value, 'unit': 's'}
                properties['u_currents'] = {'value': self.ship_params_per_step.u_currents[i].value, 'unit': 'm/s'}
                properties['v_currents'] = {'value': self.ship_params_per_step.v_currents[i].value, 'unit': 'm/s'}
                properties['u_wind_speed'] = {'value': self.ship_params_per_step.u_wind_speed[i].value, 'unit': 'm/s'}
                properties['v_wind_speed'] = {'value': self.ship_params_per_step.v_wind_speed[i].value, 'unit': 'm/s'}
                properties['pressure'] = {'value': self.ship_params_per_step.pressure[i].value, 'unit': 'Pa'}
                properties['air_temperature'] = {'value': self.ship_params_per_step.air_temperature[i].value,
                                                 'unit': '°C'}
                properties['salinity'] = {'value': self.ship_params_per_step.salinity[i].value, 'unit': '-'}
                properties['water_temperature'] = {'value': self.ship_params_per_step.water_temperature[i].value,
                                                   'unit': '°C'}
                properties['status'] = {'value': self.ship_params_per_step.status[i]}
                properties['message'] = {'value': self.ship_params_per_step.message[i]}

            feature['type'] = 'Feature'
            feature['geometry'] = geometry
            feature['properties'] = properties
            feature['id'] = i

            feature_list.append(feature)

        rp_dict['features'] = feature_list

        with open(filename, 'w') as file:
            json.dump(rp_dict, file, cls=NumpyArrayEncoder, indent=4)

    @classmethod
    def from_file(cls, filename):
        """
        Create a RouteParams object from a GeoJSON file.

        The start times need to have the format ``'%Y-%m-%d %H:%M:%S'``. The values of the ship parameters are
        converted to SI units (power in W, fuel rate in kg/s) and the values of the last point are dropped. The
        distances per step are calculated from the coordinates; courses are not read and set to -99. The route type
        is set to ``'read_from_file'`` and ``gcr`` to -99.

        :param filename: path of the GeoJSON file
        :type filename: str
        :return: route read from file
        :rtype: RouteParams
        """
        with open(filename) as file:
            rp_dict = json.load(file)

        point_list = rp_dict['features']
        count = len(point_list)

        logger.info('Reading ' + str(count) + ' coordinate pairs from file')

        lats_per_step = np.full(count, -99.)
        lons_per_step = np.full(count, -99.)
        start_time_per_step = np.full(count, datetime.now())
        speed = np.full(count, -99.)
        power = np.full(count, -99.)
        fuel_rate = np.full(count, -99.)
        rpm = np.full(count, -99.)
        r_wind = np.full(count, -99.)
        r_calm = np.full(count, -99.)
        r_waves = np.full(count, -99.)
        r_shallow = np.full(count, -99.)
        r_roughness = np.full(count, -99.)
        wave_height = np.full(count, -99.)
        wave_direction = np.full(count, -99.)
        wave_period = np.full(count, -99.)
        u_currents = np.full(count, -99.)
        v_currents = np.full(count, -99.)
        u_wind_speed = np.full(count, -99.)
        v_wind_speed = np.full(count, -99.)
        pressure = np.full(count, -99.)
        air_temperature = np.full(count, -99.)
        salinity = np.full(count, -99.)
        water_temperature = np.full(count, -99.)
        course_per_step = np.full(count - 1, -99.)
        status = np.full(count, -99)
        message = np.full(count, "")
        fuel_type = np.full(count, "")

        for ipoint in range(0, count):
            coord_pair = point_list[ipoint]['geometry']['coordinates']
            lats_per_step[ipoint] = coord_pair[1]
            lons_per_step[ipoint] = coord_pair[0]

            property = point_list[ipoint]['properties']
            start_time_per_step[ipoint] = datetime.strptime(property['time'], '%Y-%m-%d %H:%M:%S')
            speed[ipoint] = property['speed']['value']
            power[ipoint] = property['engine_power']['value']
            fuel_rate[ipoint] = property['fuel_consumption']['value']
            fuel_type[ipoint] = property['fuel_type']
            rpm[ipoint] = property['propeller_revolution']['value']

            r_wind[ipoint] = property['wind_resistance']['value']
            r_calm[ipoint] = property['calm_resistance']['value']
            r_waves[ipoint] = property['wave_resistance']['value']
            r_shallow[ipoint] = property['shallow_water_resistance']['value']
            r_roughness[ipoint] = property['hull_roughness_resistance']['value']
            wave_height[ipoint] = property['wave_height']['value']
            wave_direction[ipoint] = property['wave_direction']['value']
            wave_period[ipoint] = property['wave_period']['value']
            u_currents[ipoint] = property['u_currents']['value']
            v_currents[ipoint] = property['v_currents']['value']
            u_wind_speed[ipoint] = property['u_wind_speed']['value']
            v_wind_speed[ipoint] = property['v_wind_speed']['value']
            pressure[ipoint] = property['pressure']['value']
            air_temperature[ipoint] = property['air_temperature']['value']
            salinity[ipoint] = property['salinity']['value']
            water_temperature[ipoint] = property['water_temperature']['value']
            status[ipoint] = property['status']['value']
        #    message[ipoint] = property['message']['value']

        speed = speed[:-1] * u.meter / u.second
        power = (power[:-1] * u.kiloWatt).to(u.Watt)
        fuel_rate = (fuel_rate[:-1] * u.tonne / u.hour).to(u.kg / u.second)
        rpm = rpm[:-1] * 1 / u.minute
        r_wind = r_wind[:-1] * u.newton
        r_calm = r_calm[:-1] * u.newton
        r_waves = r_waves[:-1] * u.newton
        r_shallow = r_shallow[:-1] * u.newton
        r_roughness = r_roughness[:-1] * u.newton
        wave_height = wave_height[:-1] * u.meter
        wave_direction = wave_direction[:-1] * u.radian
        wave_period = wave_period[:-1] * u.second
        u_currents = u_currents[:-1] * u.meter / u.second
        v_currents = v_currents[:-1] * u.meter / u.second
        u_wind_speed = u_wind_speed[:-1] * u.meter / u.second
        v_wind_speed = v_wind_speed[:-1] * u.meter / u.second
        pressure = pressure[:-1] * u.kg / u.meter / u.second ** 2
        air_temperature = air_temperature[:-1] * u.deg_C
        salinity = salinity[:-1] * u.dimensionless_unscaled
        water_temperature = water_temperature[:-1] * u.deg_C

        start = (lats_per_step[0], lons_per_step[0])
        finish = (lats_per_step[count - 1], lons_per_step[count - 1])
        gcr = -99
        route_type = 'read_from_file'
        time = start_time_per_step[count - 1] - start_time_per_step[0]
        dists_per_step = cls.get_dist_from_coords(cls, lats_per_step, lons_per_step)
        dists_per_step = dists_per_step[:-1]

        count = count - 1

        ship_params_per_step = ShipParams(
            fuel_rate=fuel_rate,
            power=power,
            rpm=rpm,
            speed=speed,
            r_wind=r_wind,
            r_calm=r_calm,
            r_waves=r_waves,
            r_shallow=r_shallow,
            r_roughness=r_roughness,
            wave_height=wave_height,
            wave_direction=wave_direction,
            wave_period=wave_period,
            u_currents=u_currents,
            v_currents=v_currents,
            u_wind_speed=u_wind_speed,
            v_wind_speed=v_wind_speed,
            pressure=pressure,
            air_temperature=air_temperature,
            salinity=salinity,
            water_temperature=water_temperature,
            status=status,
            message=message
        )

        return cls(count=count, start=start, finish=finish, gcr=gcr, route_type=route_type, time=time,
                   lats_per_step=lats_per_step, lons_per_step=lons_per_step, course_per_step=course_per_step,
                   dists_per_step=dists_per_step, starttime_per_step=start_time_per_step,
                   ship_params_per_step=ship_params_per_step)

    def get_dist_from_coords(self, lats, lons):
        """
        Calculate the great circle distances between consecutive coordinate pairs.

        :param lats: latitudes in degrees
        :type lats: numpy.ndarray
        :param lons: longitudes in degrees, same length as ``lats``
        :type lons: numpy.ndarray
        :return: distance from each point to the following one in m, same length as ``lats``; the last value is -99
        :rtype: astropy.units.Quantity
        """
        nsteps = len(lats)
        dist = np.full(nsteps, -99.)

        for i in range(0, nsteps - 1):
            dist_step = geod.inverse([lats[i]], [lons[i]], [lats[i + 1]], [lons[i + 1]])
            dist[i] = dist_step['s12'][0]
        return dist * u.meter

    def plot_route(self, ax, colour, label, linestyle=False):
        """
        Plot the route and markers for start and destination on a map.

        :param ax: axes with a cartopy projection
        :type ax: cartopy.mpl.geoaxes.GeoAxes
        :param colour: matplotlib colour of the route and the markers
        :type colour: str
        :param label: legend label of the route
        :type label: str
        :param linestyle: matplotlib line style; if False, the default line style is used
        :type linestyle: str or bool
        :return: axes with the route
        :rtype: cartopy.mpl.geoaxes.GeoAxes
        """
        input_crs = ccrs.PlateCarree()
        lats = self.lats_per_step
        lons = self.lons_per_step

        if linestyle:
            ax.plot(lons, lats, color=colour, label=label, linewidth=4, linestyle=linestyle, transform=input_crs)
        else:
            ax.plot(lons, lats, color=colour, label=label, linewidth=4, transform=input_crs)

        ax.plot(self.start[1], self.start[0], marker="o", markerfacecolor=colour, markeredgecolor=colour, markersize=10)
        ax.plot(self.finish[1], self.finish[0], marker="o", markerfacecolor=colour, markeredgecolor=colour,
                markersize=10)
        return ax

    def get_power_type(self, power_type, norm=1.):
        if power_type == 'power':
            power = self.ship_params_per_step.get_power() / norm * 1. / 1000
            return {"value": power, "label": 'power consumption (kW)', "unit": u.Watt}
        if power_type == 'relative_power':
            norm = norm * u.Watt
            power = self.ship_params_per_step.get_power() / norm * 100
            return {"value": power, "label": 'power consumption (%)', "unit": u.Watt}
        if power_type == 'fuel':
            power = self.get_fuel_per_dist()
            return {"value": power, "label": "fuel consumption", "unit": u.kg}
        raise NotImplementedError(f'The power type "{power_type}" is not implemented.')

    def plot_power_vs_dist(self, icolor, label, power_type, ax, norm=1.):
        """
        Plot power or fuel consumption per step against the travel distance.

        Power is plotted as step function, fuel as bar chart in t/km. The x axis shows the travel distance in km.
        Labels are set on the current pyplot figure.

        :param icolor: index of the colour and line style from ``utils.graphics``
        :type icolor: int
        :param label: legend label
        :type label: str
        :param power_type: ``'power'``, ``'relative_power'`` or ``'fuel'``, see ``get_power_type``
        :type power_type: str
        :param ax: axes to plot on
        :type ax: matplotlib.axes.Axes
        :param norm: normalisation of the power, see ``get_power_type``
        :type norm: float
        """
        power = self.get_power_type(power_type, norm)
        dist = self.dists_per_step

        hist_values = graphics.get_hist_values_from_widths(dist, power["value"], power_type)
        hist_values["bin_contents"] = hist_values["bin_contents"]

        if power_type == 'power' or power_type == 'relative_power':
            plt.ylabel(power["label"])

            lower_bin_boundaries = (hist_values["bin_centres"] - 0.5 * hist_values["bin_widths"]) / 1000
            plt.step(
                lower_bin_boundaries,
                hist_values["bin_contents"],
                where='mid',
                linewidth=2,
                color=graphics.get_str_color(icolor),
                linestyle=graphics.get_linestyle(icolor),
                label=label
            )

            left, right = plt.xlim()
            ax.set_xlim(-100, right)
        else:
            plt.ylabel(power["label"] + ' (t/km)')
            # plt.ylabel(power["label"])
            plt.bar(
                hist_values["bin_centres"].to(u.km).value,
                hist_values["bin_contents"].to(u.tonne / u.kilometer).value,
                hist_values["bin_widths"].to(u.km).value,
                fill=False,
                color=graphics.get_colour(icolor),
                edgecolor=graphics.get_colour(icolor),
                linestyle=graphics.get_linestyle(icolor),
                label=label
            )
            ax.set_ylim(0, 0.05)
        plt.xlabel('travel distance (km)')
        plt.xticks()

    def plot_power_vs_dist_resistances(self, color, label, power_type, ax1, ax2, ax3, ax4):
        """
        Plot power or fuel consumption and the wind, wave and calm water resistance against the travel distance.

        All quantities are plotted as bar charts with the step lengths as bar widths; the x axis shows the travel
        distance in km.

        :param color: matplotlib colour
        :type color: str
        :param label: legend label
        :type label: str
        :param power_type: ``'power'`` (plotted in kW) or ``'fuel'`` (plotted in t/km), see ``get_power_type``
        :type power_type: str
        :param ax1: axes for power or fuel consumption
        :type ax1: matplotlib.axes.Axes
        :param ax2: axes for the wind resistance in N
        :type ax2: matplotlib.axes.Axes
        :param ax3: axes for the wave resistance in N
        :type ax3: matplotlib.axes.Axes
        :param ax4: axes for the calm water resistance in N
        :type ax4: matplotlib.axes.Axes
        """
        power = self.get_power_type(power_type)
        dist = self.dists_per_step
        r_wind = self.ship_params_per_step.r_wind
        r_wave = self.ship_params_per_step.r_waves
        r_calm = self.ship_params_per_step.r_calm

        hist_values = graphics.get_hist_values_from_widths(dist, power["value"], power_type)

        # only for power: also plot bin showing weighted mean. This does not make sense for fuel.
        if power_type == 'power':
            ax1.set_label(power["label"])
            ax1.bar(
                hist_values["bin_centres"].to(u.km).value,
                hist_values["bin_contents"].to(u.kiloWatt).value,
                hist_values["bin_widths"].to(u.km).value,
                fill=False, color=color, edgecolor=color, label=label, linewidth=2)

            left, right = plt.xlim()
            ax1.set_xlim(-100, right)
        else:
            ax1.set_label(power["label"] + ' (t/km)')
            ax1.bar(
                hist_values["bin_centres"].to(u.km).value,
                hist_values["bin_contents"].to(u.tonne / u.kilometer).value,
                hist_values["bin_widths"].to(u.km).value,
                fill=False, color=color, edgecolor=color, label=label
            )
            ax1.set_ylim(0, 0.05)
        ax2.bar(
            hist_values["bin_centres"].to(u.km).value,
            r_wind.value,
            hist_values["bin_widths"].to(u.km).value,
            fill=False, color=color, edgecolor=color, label=label, linewidth=2)
        ax2.set_ylabel("r_wind (N)")
        ax3.bar(
            hist_values["bin_centres"].to(u.km).value,
            r_wave.value,
            hist_values["bin_widths"].to(u.km).value,
            fill=False, color=color, edgecolor=color, label=label, linewidth=2)
        ax3.set_ylabel("r_wave (N)")
        ax4.bar(
            hist_values["bin_centres"].to(u.km).value,
            r_calm.value,
            hist_values["bin_widths"].to(u.km).value,
            fill=False, color=color, edgecolor=color, label=label, linewidth=2)
        ax4.set_ylabel("r_calm (N)")
        plt.xlabel('travel distance (km)')
        plt.xticks()

    def plot_speed_vs_dist(self, icolor, label, ax):
        """
        Plot the speed per step in m/s as step function against the travel distance in km.

        :param icolor: index of the colour and line style from ``utils.graphics``
        :type icolor: int
        :param label: legend label
        :type label: str
        :param ax: axes to plot on
        :type ax: matplotlib.axes.Axes
        """
        speed = self.ship_params_per_step.get_speed()
        dist = self.dists_per_step

        hist_values = graphics.get_hist_values_from_widths(dist, speed, "speed")
        plt.ylabel("speed (m/s)")
        lower_bin_boundaries = (hist_values["bin_centres"] - 0.5 * hist_values["bin_widths"]) / 1000

        plt.step(
            lower_bin_boundaries,
            hist_values["bin_contents"].to(u.m / u.second).value,
            where='mid',
            linewidth=2,
            color=graphics.get_str_color(icolor),
            linestyle=graphics.get_linestyle(icolor),
            label=label
        )

        left, right = plt.xlim()
        ax.set_xlim(-100, right)

        plt.xlabel('travel distance (km)')
        plt.xticks()

    # TODO check whether correct: Why do we see steps and no smooth curve?
    def plot_acc_power_vs_dist(self, color, label, power_type):
        """
        Plot the accumulated power or fuel consumption against the travel distance on the current pyplot figure.

        The values per step are multiplied with the step lengths and summed up along the route. Fuel is plotted in t.

        :param color: matplotlib colour
        :type color: str
        :param label: legend label
        :type label: str
        :param power_type: ``'power'`` or ``'fuel'``, see ``get_power_type``
        :type power_type: str
        """
        power = self.get_power_type(power_type)
        dist = self.dists_per_step

        hist_values = graphics.get_hist_values_from_widths(dist, power["value"], power_type)

        acc_bin_content = np.array([]) * power["unit"]
        fuel_sum = 0 * power["unit"]
        for i in range(0, len(hist_values["bin_centres"])):
            fuel_sum = fuel_sum + hist_values["bin_widths"][i] * hist_values["bin_contents"][i]

            acc_bin_content = np.append(acc_bin_content, fuel_sum)

        plt.xlabel('Weglänge (km)')
        if power_type == 'power':
            plt.ylabel(power["label"] + ' (' + power["unit"] + ')')
            plt.bar(
                hist_values["bin_centres"].to(u.km).value,
                acc_bin_content.to(u.kiloWatt).value,
                hist_values["bin_widths"].to(u.km).value,
                fill=False, color=color, edgecolor=color, label=label, linewidth=1
            )
        else:
            plt.ylabel('akkumulierter ' + power["label"] + ' (t)')
            plt.bar(
                hist_values["bin_centres"].to(u.km).value,
                acc_bin_content.to(u.tonne).value,
                hist_values["bin_widths"].to(u.km).value,
                fill=False, color=color, edgecolor=color, label=label, linewidth=1)
        plt.xticks()

    def plot_power_vs_dist_ratios(self, denominator, color, label, power_type):
        """
        Plot the ratio of power or fuel consumption of this route and another route against the travel distance.

        If the histograms of the two routes have a different binning, the values of ``denominator`` are linearly
        interpolated to the bin centres of this route. Plotted on the current pyplot figure.

        :param denominator: route whose values are used as denominator
        :type denominator: RouteParams
        :param color: matplotlib colour
        :type color: str
        :param label: legend label
        :type label: str
        :param power_type: ``'power'``, ``'relative_power'`` or ``'fuel'``, see ``get_power_type``
        :type power_type: str
        """
        power_nom = self.get_power_type(power_type)
        dist_nom = self.dists_per_step
        hist_values_nom = graphics.get_hist_values_from_widths(dist_nom, power_nom["value"], power_type)

        power_denom = denominator.get_power_type(power_type)
        dist_denom = denominator.dists_per_step
        hist_values_denom = graphics.get_hist_values_from_widths(dist_denom, power_denom["value"], power_type)

        if not np.array_equal(hist_values_denom["bin_centres"], hist_values_nom["bin_centres"]):
            logger.warning("Histograms for building the ratio have different binning! Need to interpolate here ...")

            hist_values_denom['bin_contents'] = np.interp(
                hist_values_nom['bin_centres'],
                hist_values_denom['bin_centres'],
                hist_values_denom['bin_contents']
            )

        hist_values_ratios = hist_values_nom["bin_contents"] / hist_values_denom["bin_contents"]
        # mean_dev = hist_values_ratios.mean()

        plt.plot(hist_values_nom["bin_centres"].to(u.km).value, hist_values_ratios,
                 marker='o', color=color, linewidth=0, label=label)
        plt.errorbar(x=hist_values_nom["bin_centres"].to(u.km).value, y=hist_values_ratios, yerr=None,
                     xerr=hist_values_nom["bin_widths"].to(u.km).value / 2, fmt=' ', color=color, linestyle=None)
        # plt.axhline(y=mean_dev, color=color, linestyle='dashed')

        plt.xlabel('travel distance (km)')
        plt.ylabel(power_nom["label"] + ' scipy shortest/model')
        plt.xticks()

    def plot_power_vs_coord(self, ax, color, label, coordstring, power_type):
        """
        Plot power or fuel consumption per step against the latitude or longitude at the start of each step.

        :param ax: axes to plot on
        :type ax: matplotlib.axes.Axes
        :param color: matplotlib colour
        :type color: str
        :param label: not used; the label is replaced by the name of the coordinate
        :type label: str
        :param coordstring: ``'lat'`` for latitude, any other value for longitude
        :type coordstring: str
        :param power_type: ``'power'`` (plotted in kW) or ``'fuel'``, see ``get_power_type``
        :type power_type: str
        """
        power = self.get_power_type(power_type)
        if coordstring == 'lat':
            coord = self.lats_per_step[:-1]
            label = 'latitude (°W)'
        else:
            coord = self.lons_per_step[:-1]
            label = 'longitude (°W)'

        if power_type == 'power':
            ax.plot(coord, power["value"].to(u.kW).value, color=color, label=label)
            plt.ylabel(power["label"] + ' (kW)')
        else:
            ax.plot(coord, power["value"].to(u.tonne).value, color=color, label=label)
            plt.ylabel(power["label"] + ' (kg)')
        plt.xlabel(label)
        # plt.ylim(1.8,2.2)
        plt.xticks()

    def get_fuel_per_dist(self):
        """
        Calculate the fuel consumed in every routing step.

        The fuel rate of each step is multiplied with the time between the start of the step and the start of the
        following step.

        :return: fuel consumed per step in kg, length ``count``
        :rtype: astropy.units.Quantity
        """
        fuel_per_second = self.ship_params_per_step.fuel_rate
        delta_time = np.full(self.count, timedelta(seconds=0))
        fuel = np.array([]) * u.kg

        for i in range(0, self.count):
            delta_time[i] = self.starttime_per_step[i + 1] - self.starttime_per_step[i]
            delta_time[i] = delta_time[i].total_seconds() * u.second
            fuel = np.append(fuel, fuel_per_second[i] * delta_time[i])

        return fuel

    def set_ship_params(self, ship_params):
        self.ship_params_per_step = ship_params

    @staticmethod
    def get_per_waypoint_coords(
            route_lons: np.ndarray,
            route_lats: np.ndarray,
            start_time: datetime,
            bs: u.Quantity
    ):
        """
        Calculate distances, courses and start times of the route segments between consecutive waypoints.

        The start time of each segment is the start time of the previous segment plus its travel time.

        :param route_lons: longitudes of the waypoints in degrees
        :type route_lons: numpy.ndarray
        :param route_lats: latitudes of the waypoints in degrees, same length as ``route_lons``
        :type route_lats: numpy.ndarray
        :param start_time: departure time at the first waypoint
        :type start_time: datetime.datetime
        :param bs: boat speed, either one value or one value per segment
        :type bs: astropy.units.Quantity
        :return: dictionary with one entry per segment (length: number of waypoints - 1) for the keys
            ``start_lats``, ``start_lons`` (degrees), ``dist`` (great circle distance in m), ``courses`` (azimuth at
            the start of the segment in degrees), ``start_times`` and ``travel_times`` (Quantity)
        :rtype: dict
        """
        debug = False
        npoints = route_lons.shape[0]
        start_lats = np.zeros(npoints - 1)
        end_lats = np.zeros(npoints - 1)
        start_lons = np.zeros(npoints - 1)
        end_lons = np.zeros(npoints - 1)

        for ipoint in range(0, npoints - 1):
            start_lats[ipoint] = route_lats[ipoint]
            start_lons[ipoint] = route_lons[ipoint]
            end_lats[ipoint] = route_lats[ipoint + 1]
            end_lons[ipoint] = route_lons[ipoint + 1]
        # ToDo: use logger.debug and args.debug
        if debug:
            print('start_lats: ', start_lats)
            print('start_lons: ', start_lons)
            print('end_lats: ', end_lats)
            print('end_lons: ', end_lons)

        move = geod.inverse(start_lats, start_lons, end_lats, end_lons)
        dist = move["s12"] * u.meter
        courses = move["azi1"] * u.degree
        travel_times = dist / bs

        start_times = np.full(npoints - 1, datetime.strptime('1970-01-01T00:00Z', '%Y-%m-%dT%H:%MZ'))
        for ipoint in range(0, npoints - 1):
            if ipoint == 0:
                start_times[ipoint] = start_time
            else:
                start_times[ipoint] = (start_times[ipoint - 1] +
                                       timedelta(seconds=travel_times[ipoint - 1].to(u.second).value))
        # ToDo: use logger.debug and args.debug
        if debug:
            print('dists: ', dist)
            print('courses: ', courses)
            print('travel_times: ', travel_times)
            print('start_times: ', start_times)

        waypoint_coors = {}
        waypoint_coors['start_lats'] = start_lats
        waypoint_coors['start_lons'] = start_lons
        waypoint_coors['dist'] = dist
        waypoint_coors['courses'] = courses
        waypoint_coors['start_times'] = start_times
        waypoint_coors['travel_times'] = travel_times
        return waypoint_coors

    def get_full_dist(self):
        dist = np.sum(self.dists_per_step)
        return dist

    def get_full_travel_time(self):
        return self.time

    def get_full_fuel(self):
        """
        Calculate the total fuel consumed on the route.

        :return: sum of the fuel consumed in every step (fuel rate times time between the start times of the steps)
            in kg
        :rtype: astropy.units.Quantity
        """
        full_fuel = 0 * u.kg
        for ipoint in range(0, self.count):
            time_passed = (self.starttime_per_step[ipoint + 1] - self.starttime_per_step[ipoint]).total_seconds()
            fuel_per_step = self.ship_params_per_step.fuel_rate[ipoint] * time_passed * u.second
            full_fuel = full_fuel + fuel_per_step

        return full_fuel

    def get_mean_power(self):
        """
        Calculate the mean engine power over the routing steps (not weighted by time or distance).

        :return: mean power in the unit of the ship parameters; ``0 kW h`` if the route has no steps
        :rtype: astropy.units.Quantity
        """
        power_sum = 0
        count = 0
        for ipoint in range(0, self.count):
            power_per_step = self.ship_params_per_step.power[ipoint]
            power_sum = power_sum + power_per_step
            count += 1
        if count > 0:
            return power_sum / count
        else:
            return 0 * u.kW * u.hour

    @staticmethod
    def from_gzip_file(filename):
        """
        Read a measured route from a gzip file.

        The file needs the columns ``Latitude``, ``Longitude``, ``SOG`` (speed over ground in knots), ``ME_LOAD``,
        ``ME_FUEL_OIL_CONSUMPTION_CALCULATED`` (kg/h), ``FORE_DRAUGHT_LEVEL``, ``AFTER_DRAUGHT_LEVEL`` and
        ``POSITION`` and a time index.

        :param filename: path of the parquet file
        :type filename: str
        :return: tuple of latitudes and longitudes in degrees, times, speed over ground in m/s (without the last
            value), fore and aft draught (pandas.Series), main engine load and fuel rate in kg/h
        :rtype: tuple
        """
        data = pandas.read_parquet(filename)
        pandas.options.display.max_columns = None
        pandas.options.display.max_rows = None

        data = data.drop(['POSITION'], axis=1)  # drop colum POSITION as it can't be converted to numeric value

        full_fuel_consumed = data['ME_FUEL_OIL_CONSUMPTION_CALCULATED'].mean() * u.kg / u.h
        mean_engine_load = data['ME_LOAD'].mean()

        # select every interval's element from dataset
        interval = 1
        sog_data = utils.unit_conversion.downsample_dataframe(data, interval)
        sog = sog_data['SOG'].values
        sog = sog[:-1] * u.Unit('knots')  # delete last element and convert from knots to m/s
        sog = sog.to(u.meter / u.second)
        power = sog_data['ME_LOAD'].values
        fuel_rate = sog_data['ME_FUEL_OIL_CONSUMPTION_CALCULATED'].values * u.kg / u.h

        # draught = 0.25 * (sog_data['AFTER_DRAUGHT_LEVEL'] + sog_data['FORE_DRAUGHT_LEVEL']
        #                   + sog_data['MIDDRAUGHT_LEVELS'] + sog_data['MIDDRAUGHT_LEVELP'])
        fore_draught = sog_data['FORE_DRAUGHT_LEVEL']
        aft_draught = sog_data['AFTER_DRAUGHT_LEVEL']

        lat = data['Latitude'][::interval].values
        lon = data['Longitude'][::interval].values

        # fix inconsistencies between pandas and numpy time formats
        time = data.index[::interval].values
        time_converted = utils.unit_conversion.convert_pandatime_to_datetime(time)

        logger.info('Reading route from file: ' + filename)
        logger.info('start: (' + str(lat[0]) + ',' + str(lon[0]) + ')')
        logger.info('start: (' + str(lat[-1]) + ',' + str(lon[-1]) + ')')
        logger.info('start time: ' + str(time[0]))
        logger.info('end time: ' + str(time[-1]))
        logger.info('mean fuel consumed (kg/h): ' + str(full_fuel_consumed))
        logger.info('ME load (percentage): ' + str(mean_engine_load))
        logger.info('fore draught: ' + str(fore_draught))
        logger.info('aft draught: ' + str(aft_draught))

        return lat, lon, time_converted, sog, fore_draught, aft_draught, power, fuel_rate

    @staticmethod
    def from_csv_file(filename):
        """
        Read a measured route from a csv file.

        The first column needs to contain the time. Further required columns are ``Latitude``, ``Longitude``,
        ``SOG`` (speed over ground in m/s), ``ME_LOAD``, ``ME_FUEL_OIL_CONSUMPTION_CALCULATED`` (kg/h),
        ``draft_fp_interpolated_between_low_speeds`` and ``draft_ap_interpolated_between_low_speeds``.

        :param filename: path of the csv file
        :type filename: str
        :return: tuple of latitudes and longitudes in degrees, times, speed over ground in m/s (without the last
            value), mean fore and mean aft draught, main engine load and fuel rate in kg/h
        :rtype: tuple
        """
        df = pandas.read_csv(filename)

        full_fuel_consumed = df['ME_FUEL_OIL_CONSUMPTION_CALCULATED'].mean() * u.kg / u.h
        mean_engine_load = df['ME_LOAD'].mean()

        sog = df["SOG"].values
        sog = sog[:-1] * u.meter / u.second

        power = df['ME_LOAD'].values
        fuel_rate = df['ME_FUEL_OIL_CONSUMPTION_CALCULATED'].values * u.kg / u.h

        fore_draught = df['draft_fp_interpolated_between_low_speeds'].mean()
        aft_draught = df['draft_ap_interpolated_between_low_speeds'].mean()

        lat = df['Latitude'].values
        lon = df['Longitude'].values

        time = df.iloc[:, 0]
        time_converted = utils.unit_conversion.convert_pandatime_to_datetime(time)

        logger.info('Reading route from file: ' + filename)
        logger.info('start: (' + str(lat[0]) + ',' + str(lon[0]) + ')')
        logger.info('start: (' + str(lat[-1]) + ',' + str(lon[-1]) + ')')
        logger.info('start time: ' + str(time_converted[0]))
        logger.info('end time: ' + str(time_converted[-1]))
        logger.info('mean fuel consumed (kg/h): ' + str(full_fuel_consumed))
        logger.info('ME load (percentage): ' + str(mean_engine_load))
        logger.info('fore draught: ' + str(fore_draught))
        logger.info('aft draught: ' + str(aft_draught))

        return lat, lon, time_converted, sog, fore_draught, aft_draught, power, fuel_rate
