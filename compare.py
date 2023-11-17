import matplotlib.pyplot as plt
import WeatherRoutingTool.utils.graphics as graphics
import WeatherRoutingTool.config as config
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.algorithms.DataUtils import *
import xarray as xr
import numpy as np

if __name__ == "__main__":

    def shipParamsPerDist(routeparams):
        fuels = np.zeros(10)
        powers = np.zeros(10)

        total_distance = np.sum(routeparams.dist_per_step)
        segment_dist = 0.1 * total_distance

        dist = 0
        segment = 0
        shipParams = routeparams.ship_params_per_step

        for i in range(len(1, routeparams.lats_per_step - 1)):
            dist += routeparams.dist_per_step[i]
            if dist <= segment_dist:
                fuels[segment] += (shipParams.fuel[i - 1])  # * routeparams.dist_per_step[i])
                powers[segment] += shipParams.power[i - 1]
            else:
                dist = dist - segment_dist
                remaining_dist = routeparams.dist_per_step[i] - dist
                if remaining_dist > 0:
                    fuels[segment] += (shipParams.fuel[i - 1])  # *remaining_dist
                    powers[segment] += shipParams.power[i - 1]
                segment += 1
                fuels[segment] += (shipParams.fuel[i - 1])  # *dist
                powers[segment] += shipParams.power[i - 1]
        return fuels, powers


def plotShipParamsHist(route_params1, route_params2, figure_path):
    fuels1, powers1 = shipParamsPerDist(route_params1)
    fuels2, powers2 = shipParamsPerDist(route_params2)
    print("Show...")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(fuels1, bins=10, alpha=0.5, label='Route 1', color='b')
    plt.hist(fuels2, bins=10, alpha=0.5, label='Route 2', color='g')
    plt.xlabel('Fuel Consumption')
    plt.ylabel('Frequency')
    plt.title('Fuel Consumption Histogram Comparison')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(powers1, bins=10, alpha=0.5, label='Route 1', color='b')
    plt.hist(powers2, bins=10, alpha=0.5, label='Route 2', color='g')
    plt.xlabel('Power Consumption')
    plt.ylabel('Frequency')
    plt.title('Power Consumption Histogram Comparison')
    plt.legend()

    plt.tight_layout()
    plt.savefig(figure_path + '/ship_params_hist.png')

    def get_closest(array, value):
        return np.abs(array - value).argmin()

    def getBase():
        file = config.WEATHER_DATA
        data = xr.open_dataset(file)

        lat1, lon1, lat2, lon2 = config.DEFAULT_MAP
        # print(lat1, lon1, lat2, lon2)

        lon_min = lon1 - 2 if lon1 < lon2 else lon2 - 2
        lon_max = lon2 + 2 if lon1 < lon2 else lon1 + 2
        lat_min = lat1 - 2 if lat1 < lat2 else lat2 - 2
        lat_max = lat2 + 2 if lat1 < lat2 else lat1 + 2

        lon_min = get_closest(data.latitude.data, lon_min)
        lon_max = get_closest(data.longitude.data, lon_max)
        lat_min = get_closest(data.latitude.data, lat_min)
        lat_max = get_closest(data.latitude.data, lat_max)

        wave_height = data.VHM0.isel(time=8)
        # print(wave_height)

        return wave_height

    def plotRoutes(routeparams, figure_path):

        lat_start, lon_start, lat_end, lon_end = config.DEFAULT_ROUTE
        start = (lat_start, lon_start)
        finish = (lat_end, lon_end)

        base = getBase()

        route_colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        plt.figure(figsize=(14, 7))
        base.plot(robust=True, aspect=2, size=7)
        for i, route in enumerate(routeparams):
            # print(route.lons_per_step, route.lats_per_step)
            color = route_colors[i]
            plt.plot(route.lats_per_step, route.lons_per_step, f"{color}-")
        plt.plot(start[1], start[0], "go", markersize=20)
        plt.plot(finish[1], finish[0], "ro", markersize=20)
        plt.savefig(figure_path + '/british_channel.png')

    def lengthHist(routeparams, figure_path):

        plt.figure(figsize=(10, 6))
        colors = ['green', 'red', 'blue']

        for i, route in enumerate(routeparams):
            plt.hist(route.dists_per_step, bins=20, alpha=0.5, label='Route ' + str(i) + 'Lengths', color=colors[i])
        plt.xlabel('Route Length (meters)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Route Lengths')
        plt.legend()
        plt.grid(True)
        plt.savefig(figure_path + '/dist_hist.png')

    def coursesHist(routeparams, figure_path):

        plt.figure(figsize=(10, 6))
        colors = ['green', 'red', 'blue']

        for i, route in enumerate(routeparams):
            plt.hist(calculate_course_for_lat_lon(route.lats_per_step, route.lons_per_step), bins=20, alpha=0.5,
                     label='Route ' + str(i) + 'Lengths', color=colors[i])
        plt.xlabel('Route Length (meters)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Route Lengths')
        plt.legend()
        plt.grid(True)
        plt.savefig(figure_path + '/course_hist1.png')

    def plotShipParams(routeparams, figure_path):
        attrs = ['fuel', 'power', 'speed', 'rpm']

        fig, axes = plt.subplots(len(attrs), 1, figsize=(8, 6), sharex=True)

        for j, route in enumerate(routeparams):
            ship_params = route.ship_params_per_step
            # print(ship_params)
            fuels = ship_params.fuel
            # print(fuels)
            powers = ship_params.power
            rpms = ship_params.rpm
            speeds = ship_params.speed

            # List of attributes to compare
            attributes = [fuels, powers, rpms, speeds]
            for i, attr in enumerate(attributes):
                ax = axes[i]
                ax.plot(range(0, route.count), attr, marker='o', linestyle='-', label='Route ' + str(j))
                ax.set_ylabel(attrs[i].capitalize())
                ax.grid(True)
                ax.legend()

        axes[-1].set_xlabel('Waypoints')
        plt.tight_layout()
        plt.savefig(figure_path + '/ship_params.png')

    # simulation study plot
    print("Hello")
    filename4 = "/Users/parichay/Mari/MariGeoRoute/GARoutes/british_channel/GA_Storm_2.json"
    filename5 = '/Users/parichay/Mari/MariGeoRoute/GARoutes/british_channel/min_time_route_230622_09.json'

    figure_path = '/Users/parichay/Mari/MariGeoRoute/WeatherRoutingTool/Figure'  # config.FIGURE_PATH
    route4 = RouteParams.from_file(filename4)
    route_iso = RouteParams.from_file(filename5)

    print(route_iso.ship_params_per_step.get_full_fuel(), route4.ship_params_per_step.get_full_fuel())

    # SplotRoutes([route3,route4,route_iso], figure_path)
    coursesHist([route4, route_iso], figure_path)
    plotShipParamsHist(route4, route_iso, figure_path)  # plotShipParams([route1,route2], figure_path)
