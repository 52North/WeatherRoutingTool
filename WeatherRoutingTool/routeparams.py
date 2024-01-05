import json
from datetime import datetime, timedelta

import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas
from geovectorslib import geod
from matplotlib import gridspec

import WeatherRoutingTool.utils as utils
import WeatherRoutingTool.utils.graphics as graphics
import WeatherRoutingTool.utils.formatting as form
from WeatherRoutingTool.utils.formatting import NumpyArrayEncoder
from WeatherRoutingTool.ship.shipparams import ShipParams

logger = logging.getLogger('WRT.Routeparams')


##
# Container class for route parameters

class RouteParams():
    count: int  # routing step (starting from 0)
    start: tuple  # lat, lon at start point (0 - 360°)
    finish: tuple  # lat, lon of destination (0 - 360°)
    gcr: tuple  # distance from start to end on great circle
    route_type: str  # route name
    time: timedelta  # time needed for the route (h)

    ship_params_per_step: ShipParams  # ship parameters per routing step
    lats_per_step: tuple  # latitude at beginning of each step + latitude destination (0-360°)
    lons_per_step: tuple  # longitude at beginning of each step + longitude destination (0-360°)
    azimuths_per_step: tuple  # azimuth per step (0-360°)
    dists_per_step: tuple  # distance traveled on great circle for every step (m)
    starttime_per_step: tuple  # start time at beginning of each step + time when destination is reached (h)

    def __init__(self, count, start, finish, gcr, route_type, time, lats_per_step, lons_per_step, azimuths_per_step,
                 dists_per_step, starttime_per_step, ship_params_per_step):
        self.count = count
        self.start = start
        self.finish = finish
        self.gcr = gcr
        self.route_type = route_type
        self.time = time
        self.lats_per_step = lats_per_step
        self.lons_per_step = lons_per_step
        self.azimuths_per_step = azimuths_per_step
        self.dists_per_step = dists_per_step
        self.starttime_per_step = starttime_per_step
        self.ship_params_per_step = ship_params_per_step

    def print_route(self):
        form.print_line()
        logger.info('Printing route:  ' + str(self.route_type))
        logger.info('Going from', self.start)
        logger.info('to', self.finish)
        logger.info('number of routing steps: ' + str(self.count))
        logger.info('latitude at start of each step: ' + str(self.lats_per_step))
        logger.info('longitude at start of each step: ' + str(self.lons_per_step))
        logger.info('azimuths for each step: ' + str(self.azimuths_per_step))
        logger.info('gcr traveled per step (m): ' + str(self.dists_per_step))
        logger.info('time at start of each step: ' + str(self.starttime_per_step))

        self.ship_params_per_step.print()

        logger.info('full fuel consumed (kg): ' + str(self.get_full_fuel('kg')))
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
        if not (np.array_equal(self.fuel, route2.fuel)):
            raise ValueError('Route fuel not matching: self=' + str(self.fuel) + ' other=' + str(route2.fuel))
        if not (np.array_equal(self.rpm, route2.rpm)):
            raise ValueError('Route rpm not matching')
        if not (np.array_equal(self.lats_per_step, route2.lats_per_step)):
            raise ValueError('Route lats_per_step not matching')
        if not (np.array_equal(self.lons_per_step, route2.lons_per_step)):
            raise ValueError('Route lons_per_step not matching')
        if not (np.array_equal(self.azimuths_per_step, route2.azimuths_per_step)):
            raise ValueError('Route azimuths_per_step not matching')
        if not (np.array_equal(self.dists_per_step, route2.dists_per_step)):
            raise ValueError('Route dists_per_step not matching')

        return bool_equal

    def convert_to_dict(self):
        rp_dict = {"count": self.count, "start": self.start, "finish": self.finish, "route type": self.route_type,
                   "gcr": self.gcr, "time": self.time, "lats_per_step": self.lats_per_step,
                   "lons_per_step": self.lons_per_step, "azimuths_per_step": self.azimuths_per_step,
                   "dists_per_step": self.dists_per_step, "starttime_per_step": self.starttime_per_step}
        return rp_dict

    def write_to_file(self, filename):
        rp_dict = self.convert_to_dict()
        with open(filename, 'w') as file:
            json.dump(rp_dict, file, cls=NumpyArrayEncoder, indent=4)

    def return_route_to_API(self, filename):
        rp_dict = {}
        rp_dict['type'] = 'FeatureCollection'
        feature_list = []

        logger.info('Write route parameters to ' + filename)

        for i in range(0, self.count + 1):
            feature = {}
            geometry = {}
            properties = {}

            geometry['type'] = 'Point'
            # geometry['coordinates'] = [self.lats_per_step[i], self.lons_per_step[i]]
            geometry['coordinates'] = [self.lons_per_step[i], self.lats_per_step[i]]

            properties['time'] = self.starttime_per_step[i]
            if i == self.count:
                properties['speed'] = {'value': -99, 'unit': 'm/s'}
                properties['engine_power'] = {'value': -99, 'unit': 'kW'}
                properties['fuel_consumption'] = {'value': -99, 'unit': 'mt/h'}
                properties['fuel_type'] = self.ship_params_per_step.fuel_type
                properties['propeller_revolution'] = {'value': -99, 'unit': 'Hz'}
                properties['calm_resistance'] = {'value': -99, 'unit': 'N'}
                properties['wind_resistance'] = {'value': -99, 'unit': 'N'}
                properties['wave_resistance'] = {'value': -99, 'unit': 'N'}
                properties['shallow_water_resistance'] = {'value': -99, 'unit': 'N'}
                properties['hull_roughness_resistance'] = {'value': -99, 'unit': 'N'}
            else:
                time_passed = (self.starttime_per_step[i + 1] - self.starttime_per_step[i]).seconds / 3600
                properties['speed'] = {'value': self.ship_params_per_step.speed[i], 'unit': 'm/s'}
                properties['engine_power'] = {'value': self.ship_params_per_step.power[i] / 1000, 'unit': 'kW'}
                properties['fuel_consumption'] = {'value': self.ship_params_per_step.fuel[i] / (time_passed * 1000),
                                                  'unit': 'mt/h'}
                properties['fuel_type'] = self.ship_params_per_step.fuel_type
                properties['propeller_revolution'] = {'value': self.ship_params_per_step.rpm[i], 'unit': 'Hz'}
                properties['calm_resistance'] = {'value': self.ship_params_per_step.r_calm[i], 'unit': 'N'}
                properties['wind_resistance'] = {'value': self.ship_params_per_step.r_wind[i], 'unit': 'N'}
                properties['wave_resistance'] = {'value': self.ship_params_per_step.r_waves[i], 'unit': 'N'}
                properties['shallow_water_resistance'] = {'value': self.ship_params_per_step.r_shallow[i], 'unit': 'N'}
                properties['hull_roughness_resistance'] = {'value': self.ship_params_per_step.r_roughness[i],
                                                           'unit': 'N'}

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
        fuel = np.full(count, -99.)
        rpm = np.full(count, -99.)
        r_wind = np.full(count, -99.)
        r_calm = np.full(count, -99.)
        r_waves = np.full(count, -99.)
        r_shallow = np.full(count, -99.)
        r_roughness = np.full(count, -99.)
        azimuths_per_step = np.full(count, -99.)
        fuel_type = np.full(count, "")

        for ipoint in range(0, count):
            coord_pair = point_list[ipoint]['geometry']['coordinates']
            lats_per_step[ipoint] = coord_pair[1]
            lons_per_step[ipoint] = coord_pair[0]

            property = point_list[ipoint]['properties']
            start_time_per_step[ipoint] = datetime.strptime(property['time'], '%Y-%m-%d %H:%M:%S')
            speed[ipoint] = property['speed']['value']
            power[ipoint] = property['engine_power']['value']
            fuel[ipoint] = property['fuel_consumption']['value']
            fuel_type[ipoint] = property['fuel_type']
            rpm[ipoint] = property['propeller_revolution']['value']

            r_wind[ipoint] = property['wind_resistance']['value']
            r_calm[ipoint] = property['calm_resistance']['value']
            r_waves[ipoint] = property['wave_resistance']['value']
            r_shallow[ipoint] = property['shallow_water_resistance']['value']
            r_roughness[ipoint] = property['hull_roughness_resistance']['value']

        start = (lats_per_step[0], lons_per_step[0])
        finish = (lats_per_step[count - 1], lons_per_step[count - 1])
        gcr = -99
        route_type = 'read_from_file'
        time = start_time_per_step[count - 1] - start_time_per_step[0]

        dists_per_step = cls.get_dist_from_coords(cls, lats_per_step, lons_per_step)

        ship_params_per_step = ShipParams(fuel=fuel, power=power, rpm=rpm, speed=speed, r_wind=r_wind, r_calm=r_calm,
                                          r_waves=r_waves, r_shallow=r_shallow, r_roughness=r_roughness)

        return cls(count=count, start=start, finish=finish, gcr=gcr, route_type=route_type, time=time,
                   lats_per_step=lats_per_step, lons_per_step=lons_per_step, azimuths_per_step=azimuths_per_step,
                   dists_per_step=dists_per_step, starttime_per_step=start_time_per_step,
                   ship_params_per_step=ship_params_per_step)

    def get_dist_from_coords(self, lats, lons):
        nsteps = len(lats)
        dist = np.full(nsteps, -99.)

        for i in range(0, nsteps - 1):
            dist_step = geod.inverse([lats[i]], [lons[i]], [lats[i + 1]], [lons[i + 1]])
            dist[i] = dist_step['s12']
        return dist

    def plot_route(self, ax, colour, label):
        lats = self.lats_per_step
        lons = self.lons_per_step
        ax.plot(lons, lats, color=colour, label=label, linewidth=2)

        ax.plot(self.start[1], self.start[0], marker="o", markerfacecolor=colour, markeredgecolor=colour, markersize=10)
        ax.plot(self.finish[1], self.finish[0], marker="o", markerfacecolor=colour, markeredgecolor=colour,
                markersize=10)
        return ax

    def get_power_type(self, power_type):
        if power_type == 'power':
            return {"value" : self.ship_params_per_step.get_power(), "label" : 'Leistung', "unit": 'kW'}
        if power_type == 'fuel':
            return {"value" : self.get_fuel_per_dist(), "label" : "Treibstoffverbrauch", "unit": 't'}

    def plot_power_vs_dist(self, color, label, power_type):
        power = self.get_power_type(power_type)
        dist = self.dists_per_step

        dist = dist / 1000  # [m] -> [km]
        hist_values = graphics.get_hist_values_from_widths(dist, power["value"], power_type)

        plt.bar(hist_values["bin_centres"], hist_values["bin_contents"], hist_values["bin_widths"], fill=False, color=color, edgecolor=color,
                label=label)
        plt.xlabel('Weglänge (km)')
        if power_type == 'power':
            plt.ylabel(power["label"] + ' (' + power["unit"] + ')')
        else:
            plt.ylabel(power["label"] + ' (' + power["unit"] + '/km)')
        plt.xticks()

    def plot_power_vs_dist_ratios(self, denominator, color, label, power_type):
        power_nom = self.get_power_type(power_type)
        dist_nom = self.dists_per_step
        dist_nom = dist_nom / 1000  # [m] -> [km]
        hist_values_nom = graphics.get_hist_values_from_widths(dist_nom, power_nom["value"], power_type)

        power_denom = denominator.get_power_type(power_type)
        dist_denom = denominator.dists_per_step
        dist_denom = dist_denom / 1000  # [m] -> [km]
        hist_values_denom = graphics.get_hist_values_from_widths(dist_denom, power_denom["value"], power_type)

        if not np.array_equal(hist_values_denom["bin_centres"], hist_values_nom["bin_centres"]):
            raise ValueError("Ratios are only possible for same binning!")

        hist_values_ratios = hist_values_nom["bin_contents"]/hist_values_denom["bin_contents"]

        plt.plot(hist_values_denom["bin_centres"], hist_values_ratios, marker='o', color=color, linewidth=0,
                 label=label)
        plt.errorbar(x=hist_values_denom["bin_centres"], y=hist_values_ratios, yerr=None,
                     xerr=hist_values_denom["bin_widths"], fmt=' ', color=color, linestyle=None)

        plt.xlabel('Weglänge (km)')
        plt.ylabel(power_nom["label"] + ' Modifiziert/Standardwert')
        plt.xticks()

    def plot_power_vs_coord(self, ax, color, label, coordstring, power_type):
        power = self.get_power_type(power_type)
        if coordstring == 'lat':
            coord = self.lats_per_step
            label = 'latitude (°W)'
        else:
            coord = self.lons_per_step
            label = 'longitude (°W)'
        power["value"] = np.delete(power["value"], power["value"].shape[0] - 1)
        coord = np.delete(coord, coord.shape[0] - 1)

        ax.plot(coord, power["value"], color=color, label=label)
        plt.xlabel(label)
        plt.ylabel(power["label"] + ' (' + power["unit"] + ')')
        # plt.ylim(1.8,2.2)
        plt.xticks()

    def get_fuel_per_dist(self):
        fuel_per_hour = self.ship_params_per_step.fuel
        delta_time = np.full(self.count - 1, timedelta(seconds=0))
        fuel = np.full(self.count, -99.)

        for i in range(0, self.count - 1):
            delta_time[i] = self.starttime_per_step[i + 1] - self.starttime_per_step[i]
            delta_time[i] = delta_time[i].total_seconds() / (60 * 60)
            fuel[i] = fuel_per_hour[i] * delta_time[i]

        return fuel

    def set_ship_params(self, ship_params):
        self.ship_params_per_step = ship_params

    def plot_power_vs_dist_with_weather(self, data_array, label_array, n_datasets):
        if n_datasets < 1:
            raise ValueError('You should at least provide 1 dataset!')

        fig = plt.figure(figsize=(25, 15))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax_power = None
        ax_rwind = None
        list_lines = []

        for idata in range(0, n_datasets):
            color = graphics.get_colour(idata)
            label = label_array[idata]
            power = data_array[idata].get_fuel_per_dist()
            dist = data_array[idata].dists_per_step
            dist = dist / 1000  # [m] -> [km]
            normalised_power = graphics.get_hist_values_from_widths(dist, power, 'fuel')
            r_wind = data_array[idata].ship_params_per_step.get_rwind()
            if r_wind[-1] == -99:
                r_wind = r_wind[:-1]
            r_wind = r_wind / 1000

            if idata == 0:
                ax_power = plt.subplot(gs[0])
                ax_power.set_ylabel('Treibstoffverbrauch (t/km)', fontsize=20)
                ax_rwind = plt.subplot(gs[1], sharex=ax_power)
                ax_rwind.set_ylabel('Windwiderstand (kN)', fontsize=20)
                ax_rwind.set_ylim(-40, 40)
                ax_power = graphics.set_graphics_standards(ax_power)
                ax_rwind = graphics.set_graphics_standards(ax_rwind)

            line_power = ax_power.bar(normalised_power["bin_centres"], normalised_power["bin_contents"], normalised_power["bin_widths"],
                                      fill=False, color=color, edgecolor=color, label=label, linewidth=2)
            list_lines.append(line_power)

            ax_rwind.bar(normalised_power["bin_centres"], r_wind, normalised_power["bin_widths"], fill=False, color=color, edgecolor=color,
                         linewidth=2)

        # ax_power.legend((line_power, line1), ('red line', 'blue line'), loc='lower left')
        ax_rwind.axhline(y=0., color='gray', linestyle='dashed', linewidth=2)
        plt.xlabel('Weglänge (km)', fontsize=20)
        fig.legend(loc='outside upper right')

        plt.xticks()

    @staticmethod
    def get_per_waypoint_coords(route_lons, route_lats, start_time, bs):
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
        dist = move["s12"]
        courses = move["azi1"]
        travel_times = dist / bs

        start_times = np.full(npoints - 1, datetime.strptime('1970-01-01T00:00Z', '%Y-%m-%dT%H:%MZ'))
        for ipoint in range(0, npoints - 1):
            if ipoint == 0:
                start_times[ipoint] = start_time
            else:
                start_times[ipoint] = start_times[ipoint - 1] + timedelta(seconds=travel_times[ipoint - 1])
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

    def get_full_dist(self, unit='km'):
        dist = np.sum(self.dists_per_step)

        if unit == 'km':
            return dist / 1000
        if unit == 'm':
            return dist

    def get_full_travel_time(self, unit='h'):
        if unit == 'h':
            return self.time.total_seconds() / 3600
        if unit == 'min':
            return self.time.total_seconds() / 60
        if unit == 'sec':
            return self.time.total_seconds()
        if unit == 'datetime':
            return self.time

    def get_full_fuel(self, unit='t'):
        full_fuel = 0
        for ipoint in range(0, self.count - 1):
            time_passed = (self.starttime_per_step[ipoint + 1] - self.starttime_per_step[ipoint]).total_seconds() / 3600
            fuel_per_step = self.ship_params_per_step.fuel[ipoint] * time_passed

            if unit == 'kg':
                fuel_per_step = fuel_per_step * 1000
            full_fuel = full_fuel + fuel_per_step

        return full_fuel

    @classmethod
    def from_gzip_file(cls, filename):
        data = pandas.read_parquet(filename)
        data = data.drop(['POSITION'], axis=1)  # drop colum POSITION as it can't be converted to numeric value

        # select every interval's element from dataset
        interval = 10
        sog_data = utils.unit_conversion.downsample_dataframe(data, interval)
        sog= sog_data['SOG'].values
        lat = data['Latitude'][::interval].values
        lon = data['Longitude'][::interval].values

        # fix inconsistencies between pandas and numpy time formats
        time = data.index[::interval].values
        time_converted = utils.unit_conversion.convert_pandatime_to_datetime(time)

        return lat, lon, time_converted, sog
