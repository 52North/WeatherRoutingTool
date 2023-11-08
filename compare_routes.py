import matplotlib.pyplot as plt

import datetime as dt

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

    # simulation: Alexandria - Marseille
    #filename1 = ("/home/kdemmich/MariData/Simulationsstudien_NovDez23/Alexandria_Marseille/bestFOC/min_time_route.json")
    #filename2 = (
    #    "/home/kdemmich/MariData/Simulationsstudien_NovDez23/Alexandria_Marseille/bestWeather/min_time_route.json")
    #filename3 = ("/home/kdemmich/MariData/Simulationsstudien_NovDez23/Alexandria_Marseille/fastest/min_time_route.json")
    #filename4 = (
    #    "/home/kdemmich/MariData/Simulationsstudien_NovDez23/Manipulated_Routes/Alexandria_Marseille/original/min_time_route.json")

    filename1 = ("/home/kdemmich/MariData/Simulationsstudien_NovDez23/Columbo_Singapore/bestFOC/min_time_route.json")
    filename2 = (
        "/home/kdemmich/MariData/Simulationsstudien_NovDez23/Columbo_Singapore/bestWeather/min_time_route.json")
    filename3 = ("/home/kdemmich/MariData/Simulationsstudien_NovDez23/Columbo_Singapore/fastest/min_time_route.json")
    filename4 = (
        "/home/kdemmich/MariData/Code/Data/RouteCollection/min_time_route.json")

    figurefile = "/home/kdemmich/MariData/Code/Figures"

    depth_data = ""

    rp_read1 = RouteParams.from_file(filename1)
    rp_read2 = RouteParams.from_file(filename2)
    rp_read3 = RouteParams.from_file(filename3)
    rp_read4 = RouteParams.from_file(filename4)

    rp_1_str = 'best FOC'
    rp_2_str = 'best weather'
    rp_3_str = 'fastest'
    rp_4_str = 'original'

    ##
    # init wheather
    windfile = "/home/kdemmich/MariData/Simulationsstudien_NovDez23/EnvData/bbox_/indian_ocean_earlier_incl.nc"
    # British Channel
    # departure_time = "2023-06-21T12:00Z"
    # time_for_plotting = "2023-06-21T12:00Z"
    # time_forecast = 60
    # lat1, lon1, lat2, lon2 = '44', '-15', '53', '3'
    # Iceland
    departure_time = "2023-09-28T09:00Z"
    time_for_plotting = "2023-11-01T09:00Z"
    time_forecast = 60
    lat1, lon1, lat2, lon2 = (44, -15, 53, 3)

    departure_time_dt = dt.datetime.strptime(departure_time, '%Y-%m-%dT%H:%MZ')
    plot_time = dt.datetime.strptime(time_for_plotting, '%Y-%m-%dT%H:%MZ')
    default_map = Map(lat1, lon1, lat2, lon2)

    wf = WeatherFactory()
    wt = wf.get_weather("from_file", windfile, departure_time_dt, time_forecast, 3, default_map)

    fig, ax = plt.subplots(figsize=(12, 7))
    # wt.plot_weather_map(fig, ax, plot_time, "wind")

    ##
    # init Constraints
    water_depth = WaterDepth('from_file', 20, default_map, depth_data)

    ##
    # plotting routes in depth profile
    fig, ax = plt.subplots(figsize=(12, 7))
    ax = water_depth.plot_route_in_constraint(rp_read1, 0, fig, ax)
    ax = rp_read1.plot_route(ax, graphics.get_colour(0), rp_1_str)
    ax = rp_read2.plot_route(ax, graphics.get_colour(1), rp_2_str)
    ax = rp_read3.plot_route(ax, graphics.get_colour(2), rp_3_str)
    ax = rp_read4.plot_route(ax, graphics.get_colour(3), rp_4_str)
    ax.legend()
    plt.savefig(figurefile + '/route_waterdepth.png')

    ##
    # plotting power vs. distance
    fig, ax = plt.subplots(figsize=(12, 8), dpi=96)
    #rp_read1.plot_power_vs_dist(graphics.get_colour(0), rp_1_str)
    #rp_read2.plot_power_vs_dist(graphics.get_colour(1), rp_2_str)
    #rp_read3.plot_power_vs_dist(graphics.get_colour(2), rp_3_str)
    rp_read4.plot_power_vs_dist(graphics.get_colour(3), rp_4_str)

    ax.legend(loc='lower left')
    # ax.set_ylim(0, 0.016)
    plt.savefig(figurefile + '/route_power.png')

    ##
    # plotting power vs. lon
    coordstring = "lon"
    fig, ax = plt.subplots(figsize=(12, 8), dpi=96)
    rp_read1.plot_power_vs_coord(ax, graphics.get_colour(0), rp_1_str, coordstring)
    rp_read2.plot_power_vs_coord(ax, graphics.get_colour(1), rp_2_str, coordstring)
    rp_read3.plot_power_vs_coord(ax, graphics.get_colour(2), rp_3_str, coordstring)
    rp_read4.plot_power_vs_coord(ax, graphics.get_colour(3), rp_4_str, coordstring)

    ax.legend(loc='lower left')
    # ax.set_ylim(0, 0.016)
    plt.savefig(figurefile + '/route_powervs' + coordstring + '.png')

    ##
    # plotting power vs dist vs weather
    data_array = [rp_read4]
    label_array = [rp_4_str]
    rp_read1.plot_power_vs_dist_with_weather(data_array, label_array, 1)

    plt.savefig(figurefile + '/route_power_vs_dist_weather.png')

    ##
    # write full fuel

    print('Full fuel consumption:')
    print(rp_1_str + ': ' + str(rp_read1.get_full_fuel()))
    print(rp_2_str + ': ' + str(rp_read2.get_full_fuel()))
    print(rp_3_str + ': ' + str(rp_read3.get_full_fuel()))
    print(rp_4_str + ': ' + str(rp_read4.get_full_fuel()))

    print('Full travel dist:')
    print(rp_1_str + ': ' + str(rp_read1.get_full_dist()))
    print(rp_2_str + ': ' + str(rp_read2.get_full_dist()))
    print(rp_3_str + ': ' + str(rp_read3.get_full_dist()))
    print(rp_4_str + ': ' + str(rp_read4.get_full_dist()))

    print('Full travel time:')
    print(rp_1_str + ': ' + str(rp_read1.get_full_travel_time('datetime')))
    print(rp_2_str + ': ' + str(rp_read2.get_full_travel_time('datetime')))
    print(rp_3_str + ': ' + str(rp_read3.get_full_travel_time('datetime')))
    print(rp_4_str + ': ' + str(rp_read4.get_full_travel_time('datetime')))
