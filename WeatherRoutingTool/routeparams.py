import datetime
import datetime as dt
import json

import numpy as np
import matplotlib.pyplot as plt
from geovectorslib import geod

import WeatherRoutingTool.utils.graphics as graphics
import WeatherRoutingTool.utils.formatting as form
from WeatherRoutingTool.utils.formatting import NumpyArrayEncoder
from WeatherRoutingTool.ship.shipparams import ShipParams

##
# Container class for route parameters

class RouteParams():
    count: int          # routing step (starting from 0)
    start: tuple        # lat, lon at start point (0 - 360°)
    finish: tuple       # lat, lon of destination (0 - 360°)
    gcr: tuple          # distance from start to end on great circle
    route_type: str     # route name
    time: dt.timedelta  # time needed for the route (h)

    ship_params_per_step: ShipParams # ship parameters per routing step
    lats_per_step: tuple        # latitude at beginning of each step + latitude destination (0-360°)
    lons_per_step: tuple        # longitude at beginning of each step + longitude destination (0-360°)
    azimuths_per_step: tuple    # azimuth per step (0-360°)
    dists_per_step: tuple       # distance traveled on great circle for every step (m)
    starttime_per_step: tuple   # start time at beginning of each step + time when destination is reached (h)

    def __init__(self, count, start, finish, gcr,  route_type, time, lats_per_step, lons_per_step, azimuths_per_step, dists_per_step,  starttime_per_step,   ship_params_per_step):
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
        self.starttime_per_step =starttime_per_step
        self.ship_params_per_step = ship_params_per_step

    def print_route(self):
        form.print_line()
        print('Printing route:  ' + str(self.route_type))
        print('Going from', self.start)
        print('to', self.finish)
        print('number of routing steps: ' + str(self.count))
        print('latitude at start of each step: ' + str(self.lats_per_step))
        print('longitude at start of each step: ' + str(self.lons_per_step))
        print('azimuths for each step: ' + str(self.azimuths_per_step))
        print('gcr traveled per step (m): ' + str(self.dists_per_step))
        print('time at start of each step: ' + str(self.starttime_per_step))

        self.ship_params_per_step.print()

        print('full fuel consumed (kg): ' + str(self.ship_params_per_step.get_full_fuel()))
        print('full travel time (h): ' + str(self.time))
        print('travel distance on great circle (m): ' + str(self.gcr))

        form.print_line()

    def __eq__(self, route2):
        bool_equal=True
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
        rp_dict = {
            "count" : self.count,
            "start" : self.start,
            "finish": self.finish,
            "route type": self.route_type,
            "gcr": self.gcr,
            "time" : self.time,
            "lats_per_step" : self.lats_per_step,
            "lons_per_step" : self.lons_per_step,
            "azimuths_per_step" : self.azimuths_per_step,
            "dists_per_step" : self.dists_per_step,
            "starttime_per_step" : self.starttime_per_step
        }
        return rp_dict

    def write_to_file(self, filename):
        rp_dict = self.convert_to_dict()
        with open(filename, 'w') as file:
            json.dump(rp_dict, file, cls=NumpyArrayEncoder, indent=4)

    def return_route_to_API(self, filename):
        rp_dict = {}
        rp_dict['type'] = 'FeatureCollection'
        feature_list = []

        print('Writing params to ', filename)

        for i in range(0, self.count+1):
            feature = {}
            geometry = {}
            properties = {}

            geometry['type'] = 'Point'
            #geometry['coordinates'] = [self.lats_per_step[i], self.lons_per_step[i]]
            geometry['coordinates'] = [self.lons_per_step[i], self.lats_per_step[i]]

            properties['time'] = self.starttime_per_step[i]
            if i == self.count:
                properties['speed'] = {'value': -99, 'unit': 'm/s'}
                properties['engine_power'] = {'value': -99, 'unit': 'kW'}
                properties['fuel_consumption'] = {'value': -99, 'unit': 'mt/h'}
                properties['fuel_type'] = self.ship_params_per_step.fuel_type
                properties['propeller_revolution'] = {'value': -99, 'unit': 'Hz'}
            else:
                time_passed = (self.starttime_per_step[i+1]-self.starttime_per_step[i]).seconds/3600
                properties['speed'] = {'value' : self.ship_params_per_step.speed[i+1], 'unit' : 'm/s'}
                properties['engine_power'] = {'value' : self.ship_params_per_step.power[i+1]/1000, 'unit' : 'kW'}
                properties['fuel_consumption'] = {'value' : self.ship_params_per_step.fuel[i+1]/(time_passed * 1000), 'unit' : 'mt/h'}
                properties['fuel_type'] = self.ship_params_per_step.fuel_type
                properties['propeller_revolution'] = {'value' : self.ship_params_per_step.rpm[i+1], 'unit' : 'Hz'}

            feature['type'] = 'Feature'
            feature['geometry'] = geometry
            feature['property'] = properties

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

        print('Reading ' + str(count) + ' coordinate pairs from file')

        lats_per_step = np.full(count,-99.)
        lons_per_step = np.full(count,-99.)
        start_time_per_step = np.full(count, datetime.datetime.now())
        speed = np.full(count, -99.)
        power = np.full(count, -99.)
        fuel = np.full(count, -99.)
        rpm = np.full(count, -99.)
        azimuths_per_step = np.full(count, -99.)
        fuel_type = np.full(count, "")

        for ipoint in range(0,count):
            coord_pair = point_list[ipoint]['geometry']['coordinates']
            lats_per_step[ipoint] = coord_pair[0]
            lons_per_step[ipoint] = coord_pair[1]

            property = point_list[ipoint]['property']
            start_time_per_step[ipoint] = dt.datetime.strptime(property['time'], '%Y-%m-%d %H:%M:%S')
            speed[ipoint] = property['speed']['value']
            power[ipoint] = property['engine_power']['value']
            fuel[ipoint] = property['fuel_consumption']['value']
            fuel_type[ipoint] = property['fuel_type']
            rpm[ipoint] = property['propeller_revolution']['value']

        start = (lats_per_step[0],lons_per_step[0])
        finish = (lats_per_step[count-1],lons_per_step[count-1])
        gcr = -99
        route_type = 'read_from_file'
        time = start_time_per_step[count-1] - start_time_per_step[0]

        dists_per_step = cls.get_dist_from_coords(cls, lats_per_step, lons_per_step)

        ship_params_per_step = ShipParams(fuel, power, rpm, speed)

        return cls(
            count = count,
            start = start,
            finish = finish,
            gcr = gcr,
            route_type = route_type,
            time = time,
            lats_per_step = lats_per_step,
            lons_per_step = lons_per_step,
            azimuths_per_step = azimuths_per_step,
            dists_per_step = dists_per_step,
            starttime_per_step = start_time_per_step,
            ship_params_per_step = ship_params_per_step
        )

    def get_dist_from_coords(self, lats, lons):
        nsteps = len(lats)
        dist = np.full(nsteps, -99.)

        for i in range(0,nsteps-1):
            dist_step = geod.inverse([lats[i]], [lons[i]],[lats[i+1]],[lons[i+1]])
            dist[i] = dist_step['s12']
        return dist

    def plot_route(self, ax, colour, label):
        lats = self.lats_per_step
        lons = self.lons_per_step
        ax.plot(lons, lats, color = colour, label = label, linewidth=2)

        ax.plot(self.start[1], self.start[0], marker="o", markerfacecolor=colour, markeredgecolor=colour,
                markersize=10)
        ax.plot(self.finish[1], self.finish[0], marker="o", markerfacecolor=colour, markeredgecolor=colour,
                markersize=10)
        return ax

    def plot_power_vs_dist(self, color, label):
        power = self.get_fuel_per_dist()
        dist = self.dists_per_step
        lat = self.lats_per_step
        lon = self.lons_per_step

        dist = dist/1000    # [m] -> [km]
        hist_values = graphics.get_hist_values_from_widths(dist, power)

        plt.bar(hist_values["bin_centres"], hist_values["bin_content"], dist, fill=False, color = color, edgecolor = color, label = label)
        plt.xlabel('Weglänge (km)')
        #plt.ylabel('Energie (kWh/km)')
        plt.ylabel('Treibstoffverbrauch (t/km)')
        plt.xticks()

    def get_fuel_per_dist(self):
        fuel_per_hour = self.ship_params_per_step.fuel
        delta_time = np.full(self.count-1, datetime.timedelta(seconds=0))
        fuel = np.full(self.count, -99.)

        for i in range(0,self.count-1):
            delta_time[i] = self.starttime_per_step[i+1]-self.starttime_per_step[i]
            delta_time[i] = delta_time[i].total_seconds()/(60*60)
            fuel[i] = fuel_per_hour[i] * delta_time[i]

        return fuel

