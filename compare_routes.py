import matplotlib.pyplot as plt

import utils.graphics as graphics
import config
from constraints.constraints import *
from routeparams import RouteParams
from utils.maps import Map
from weather import WeatherCondFromFile

if __name__ == "__main__":
    # Intermediate waypoints plot
    # filename1 = "/home/kdemmich/MariData/Simulationsstudie_April23/Route_Thames_Bordeaux/230509_results
    # /Thames_Bordeaux_WP1_WH0.json"
    # filename2 = "/home/kdemmich/MariData/Simulationsstudie_April23/Route_Thames_Bordeaux/230509_results
    # /Thames_Bordeaux_WP3_WH8.json"

    # simulation study plot
    filename1 = "/home/kdemmich/MariData/Code/Data/RouteCollection/min_time_route.json"
    filename2 = "/home/kdemmich/MariData/Simulationsstudie_April23/Route_Thames_Bordeaux/230515_results" \
                "/Thames_Bordeaux_WP2_WH4.json"
    filename3 = "/home/kdemmich/MariData/Simulationsstudie_April23/Route_Thames_Bordeaux/230515_results" \
                "/Thames_Bordeaux_WP3_WH8.json"

    figurefile = "/home/kdemmich/MariData/Code/Figures"
    rp_read1 = RouteParams.from_file(filename1)
    rp_read2 = RouteParams.from_file(filename2)
    rp_read3 = RouteParams.from_file(filename3)

    rp_read1.print_route()
    rp_read2.print_route()

    ##
    # init wheather
    windfile = config.WEATHER_DATA
    model = config.START_TIME
    start_time = dt.datetime.strptime(config.START_TIME, '%Y%m%d%H')
    hours = config.TIME_FORECAST
    lat1, lon1, lat2, lon2 = config.DEFAULT_MAP
    depthfile = config.DEPTH_DATA
    map = Map(lat1, -10., lat2, lon2)
    wt = WeatherCondFromFile(model, start_time, hours, 3)
    wt.set_map_size(map)
    wt.read_dataset(windfile)

    ##
    # init Constraints
    water_depth = WaterDepth(config.DEPTH_DATA, 20, map)

    ##
    # plotting routes in depth profile
    fig, ax = plt.subplots(figsize=(12, 7))
    ax = water_depth.plot_route_in_constraint(rp_read1, 0, fig, ax)
    ax = rp_read1.plot_route(ax, graphics.get_colour(0), "Route WP1")
    ax = rp_read2.plot_route(ax, graphics.get_colour(3), "Route WP2")
    ax = rp_read3.plot_route(ax, graphics.get_colour(2), "Route WP3")

    # rp_read1.plot_route(ax, 'orangered', "10m Tiefgang")
    # rp_read2.plot_route(ax, 'cyan', "kein Tiefgang")

    ax.plot(-5.502222, 45.715000, marker="o", markerfacecolor=graphics.get_colour(1),
            markeredgecolor=graphics.get_colour(1), linestyle='None', markersize=10, label='Intermediate WPs')
    ax.plot(0.609062, 50.600152, marker="o", markerfacecolor=graphics.get_colour(1),
            markeredgecolor=graphics.get_colour(1), markersize=10)
    ax.plot(-4.176667, 46.923056, marker="o", markerfacecolor=graphics.get_colour(1),
            markeredgecolor=graphics.get_colour(1), markersize=10)
    ax.plot(-3.617778, 47.358611, marker="o", markerfacecolor=graphics.get_colour(1),
            markeredgecolor=graphics.get_colour(1), markersize=10)
    ax.set_xlim(-8, 2.5)
    ax.set_ylim(44, 52)
    ax.legend()
    plt.savefig(figurefile + '/route_waterdepth.png')

    ##
    # plotting power vs. distance
    fig, ax = plt.subplots(figsize=(12, 8), dpi=96)
    rp_read1.plot_power_vs_dist(graphics.get_colour(0), "gutes Wetter")
    rp_read2.plot_power_vs_dist(graphics.get_colour(1), "schlechtes Wetter")
    ax.legend()
    ax.set_ylim(0, 0.016)
    plt.savefig(figurefile + '/route_power.png')

    # plotting routes in wind data
    # fig, ax = plt.subplots(figsize=(12, 7))
    # wt.plot_weather_map(fig,ax,
    # "2023-02-08T06:00:00.000000000")  # plt.show()
