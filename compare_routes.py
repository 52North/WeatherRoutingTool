import matplotlib.pyplot as plt

import datetime as dt

import WeatherRoutingTool.config as config
import WeatherRoutingTool.utils.graphics as graphics
from WeatherRoutingTool.constraints.constraints import *
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.utils.maps import Map
from WeatherRoutingTool.weather_factory import WeatherFactory

if __name__ == "__main__":
    # Intermediate waypoints plot
    # filename1 = "/home/kdemmich/MariData/Simulationsstudie_April23/Route_Thames_Bordeaux/230509_results
    # /Thames_Bordeaux_WP1_WH0.json"
    # filename2 = "/home/kdemmich/MariData/Simulationsstudie_April23/Route_Thames_Bordeaux/230509_results
    # /Thames_Bordeaux_WP3_WH8.json"

    # simulation study plot
    filename1 = ("/home/kdemmich/MariData/Code/Data/RouteCollection/CompareWeather_231016/min_time_route_260623_18Uh"
                 ".json")
    filename2 = ("/home/kdemmich/MariData/Code/Data/RouteCollection/CompareWeather_231016/min_time_route_270623_3Uhr"
                 ".json")
    filename3 = ("/home/kdemmich/MariData/Code/Data/RouteCollection/min_time_route.json")

    figurefile = "/home/kdemmich/MariData/Code/Figures"
    rp_read1 = RouteParams.from_file(filename1)
    rp_read2 = RouteParams.from_file(filename2)
    rp_read3 = RouteParams.from_file(filename3)

    ##
    # init wheather
    windfile = "/home/kdemmich/MariData/Code/Data/WheatherFiles/2023_10_20_Iceland_long.nc"
    # British Channel
    # departure_time = "2023-06-21T12:00Z"
    # time_for_plotting = "2023-06-21T12:00Z"
    # time_forecast = 60
    # lat1, lon1, lat2, lon2 = '44', '-15', '53', '3'
    # Iceland
    departure_time = "2023-09-28T09:00Z"
    time_for_plotting = "2023-10-20T15:00Z"
    time_forecast = 60
    lat1, lon1, lat2, lon2 = '60', '-30', '69', '-8'

    departure_time_dt = dt.datetime.strptime(departure_time, '%Y-%m-%dT%H:%MZ')
    plot_time = dt.datetime.strptime(time_for_plotting, '%Y-%m-%dT%H:%MZ')
    default_map = Map(lat1, lon1, lat2, lon2)

    wf = WeatherFactory()
    wt = wf.get_weather("from_file", windfile, departure_time_dt, time_forecast, 3, default_map)

    fig, ax = plt.subplots(figsize=(12, 7))
    # wt.plot_weather_map(fig, ax, plot_time, "wind")

    ##
    # init Constraints
    water_depth = WaterDepth('from_file', 20, default_map, config.DEPTH_DATA)

    ##
    # plotting routes in depth profile
    fig, ax = plt.subplots(figsize=(12, 7))
    # ax = water_depth.plot_route_in_constraint(rp_read1, 0, fig, ax)
    ax = rp_read1.plot_route(ax, graphics.get_colour(0), "wind scenario")
    ax = rp_read2.plot_route(ax, graphics.get_colour(1), "current scenario")
    ax = rp_read3.plot_route(ax, graphics.get_colour(2), "Route travel_dist")

    # rp_read1.plot_route(ax, 'orangered', "10m Tiefgang")
    # rp_read2.plot_route(ax, 'cyan', "kein Tiefgang")

    # ax.plot(-5.502222, 45.715000, marker="o", markerfacecolor=graphics.get_colour(1),
    #        markeredgecolor=graphics.get_colour(1), linestyle='None', markersize=10, label='Intermediate WPs')
    # ax.plot(0.609062, 50.600152, marker="o", markerfacecolor=graphics.get_colour(1),
    #        markeredgecolor=graphics.get_colour(1), markersize=10)
    # ax.plot(-4.176667, 46.923056, marker="o", markerfacecolor=graphics.get_colour(1),
    #        markeredgecolor=graphics.get_colour(1), markersize=10)
    # ax.plot(-3.617778, 47.358611, marker="o", markerfacecolor=graphics.get_colour(1),
    #        markeredgecolor=graphics.get_colour(1), markersize=10)
    # ax.set_xlim(-8, 2.5)
    # ax.set_ylim(44, 52)
    ax.legend()
    plt.savefig(figurefile + '/route_waterdepth.png')

    ##
    # plotting power vs. distance
    fig, ax = plt.subplots(figsize=(12, 8), dpi=96)
    rp_read1.plot_power_vs_dist(graphics.get_colour(0), "wind scenario")
    rp_read2.plot_power_vs_dist(graphics.get_colour(1), "current scenario")
    rp_read3.plot_power_vs_dist(graphics.get_colour(2), "Route travel_dist")

    ax.legend(loc='lower left')
    # ax.set_ylim(0, 0.016)
    plt.savefig(figurefile + '/route_power.png')

    ##
    # plotting power vs. lon
    coordstring = "lon"
    fig, ax = plt.subplots(figsize=(12, 8), dpi=96)
    rp_read1.plot_power_vs_coord(ax, graphics.get_colour(0), "wind scenario", coordstring)
    rp_read2.plot_power_vs_coord(ax, graphics.get_colour(1), "current scenario", coordstring)
    rp_read3.plot_power_vs_coord(ax, graphics.get_colour(2), "good weather", coordstring)

    ax.legend(loc='lower left')
    # ax.set_ylim(0, 0.016)
    plt.savefig(figurefile + '/route_powervs' + coordstring + '.png')

    ##
    # plotting power vs dist vs weather
    data_array = [rp_read1, rp_read3]
    label_array = ['wind scenario', 'good weather']
    rp_read1.plot_power_vs_dist_with_weather(data_array, label_array, 2)
