import matplotlib.pyplot as plt

import datetime as dt

import WeatherRoutingTool.utils.graphics as graphics
from WeatherRoutingTool.constraints.constraints import *
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.utils.maps import Map
from WeatherRoutingTool.weather_factory import WeatherFactory


def plot_power_vs_dist(rp_list, rp_str_list, power_type='fuel'):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=96)
    for irp in range(0, len(rp_list)):
        rp_list[irp].plot_power_vs_dist(graphics.get_colour(irp), rp_str_list[irp], power_type)

    ax.legend(loc='lower left')
    # ax.set_ylim(0, 0.016)
    plt.savefig(figurefile + '/' + power_type + '_vs_dist.png')


def plot_power_vs_coord(rp_list, rp_str_list, coordstring, power_type='fuel'):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=96)
    for irp in range(0, len(rp_list)):
        rp_list[irp].plot_power_vs_coord(ax, graphics.get_colour(irp), rp_str_list[irp], coordstring, power_type)
    ax.legend(loc='lower left')
    # ax.set_ylim(0, 0.016)
    plt.savefig(figurefile + '/' + power_type + '_vs_' + coordstring + '.png')


def plot_power_vs_dist_ratios(rp_list, rp_str_list, power_type='fuel'):
    # windspeed = '12.5'

    fig, ax = plt.subplots(figsize=(12, 8), dpi=96)
    ax.set_ylim(0.8, 1.2)

    for irp in range(1, len(rp_list)):
        rp_list[irp].plot_power_vs_dist_ratios(rp_list[0], graphics.get_colour(irp),
                                               rp_str_list[irp] + '/' + rp_str_list[0], power_type)

    ax.legend(loc='lower left')
    # ax.set_title('Windige Wetterlage (Windgeschwindigkeit: ' + windspeed + ' m/s)')
    plt.savefig(figurefile + '/' + power_type + '_vs_dist_ratios' + '.png')


if __name__ == "__main__":
    filename1 = ("/home/kdemmich/MariData/Debug_Multiple_Routes/Routes/route_1.json")
    filename2 = ("/home/kdemmich/MariData/Debug_Multiple_Routes/Routes/route_2.json")
    filename3 = ("/home/kdemmich/MariData/IMDC_paper/Routes/route_calm_weather_105_calmwaterres.json")
    filename4 = ("/home/kdemmich/MariData/Code/Data/RouteCollection/min_time_route.json")

    figurefile = "/home/kdemmich/MariData/Debug_Multiple_Routes/Figures"

    windfile = "/home/kdemmich/MariData/Simulationsstudien_NovDez23/EnvData/bbox_/indian_ocean_earlier_incl.nc"

    depth_data = ""

    rp_read1 = RouteParams.from_file(filename1)
    rp_read2 = RouteParams.from_file(filename2)
    # rp_read3 = RouteParams.from_file(filename3)
    # rp_read4 = RouteParams.from_file(filename4)

    rp_1_str = 'Standardeinstellung'
    rp_2_str = '95% Glattwasserwiderstand'
    rp_3_str = '105% Glattwasserwiderstand'
    rp_4_str = 'original'

    rp_list = [rp_read1, rp_read2]
    rp_str_list = [rp_1_str, rp_2_str]

    do_plot_weather = False
    do_plot_route = True
    do_plot_power_vs_dist = False
    do_plot_fuel_vs_dist = False

    do_plot_power_vs_lon = False
    do_plot_fuel_vs_lon = False
    do_plot_power_vs_lat = False
    do_plot_fuel_vs_lat = False

    do_plot_power_vs_dist_showing_weather = False
    do_plot_power_vs_dist_ratios = False
    do_plot_fuel_vs_dist_ratios = False
    do_write_fuel = False

    ##
    # init wheather
    departure_time = "2023-09-28T09:00Z"
    time_for_plotting = "2023-11-01T09:00Z"
    time_forecast = 60
    lat1, lon1, lat2, lon2 = (44, -15, 53, 3)

    #############################################################################
    plt.rcParams['font.size'] = graphics.get_standard('font_size')

    departure_time_dt = dt.datetime.strptime(departure_time, '%Y-%m-%dT%H:%MZ')
    plot_time = dt.datetime.strptime(time_for_plotting, '%Y-%m-%dT%H:%MZ')
    default_map = Map(lat1, lon1, lat2, lon2)

    if do_plot_weather:
        wf = WeatherFactory()
        wt = wf.get_weather("from_file", windfile, departure_time_dt, time_forecast, 3, default_map)

        fig, ax = plt.subplots(figsize=(12, 7))
        wt.plot_weather_map(fig, ax, plot_time, "wind")

    ##
    # init Constraints
    # water_depth = WaterDepth('from_file', 20, default_map, depth_data)

    ##
    # plotting routes in depth profile
    if do_plot_route:
        fig, ax = plt.subplots(figsize=graphics.get_standard('fig_size'))
        ax.axis('off')
        ax.xaxis.set_tick_params(labelsize='large')
        fig, ax = graphics.generate_basemap(fig, None, rp_read1.start, rp_read1.finish, '', False)

        # ax = water_depth.plot_route_in_constraint(rp_read1, 0, fig, ax)
        for irp in range(0, len(rp_list)):
            ax = rp_list[irp].plot_route(ax, graphics.get_colour(irp), rp_str_list[irp])
        ax.legend()
        plt.savefig(figurefile + '/route_waterdepth.png')

    ##
    # plotting  vs. distance
    if do_plot_power_vs_dist:
        plot_power_vs_dist(rp_list, rp_str_list, 'power')

    if do_plot_fuel_vs_dist:
        plot_power_vs_dist(rp_list, rp_str_list, 'fuel')

    ##
    # plotting power vs. coordinate
    if do_plot_power_vs_lat:
        plot_power_vs_coord(rp_list, rp_str_list, 'lat', 'power')

    if do_plot_fuel_vs_lat:
        plot_power_vs_coord(rp_list, rp_str_list, 'lat', 'fuel')

    if do_plot_power_vs_lon:
        plot_power_vs_coord(rp_list, rp_str_list, 'lon', 'power')

    if do_plot_fuel_vs_lon:
        plot_power_vs_coord(rp_list, rp_str_list, 'lon', 'fuel')

    ##
    # plotting power vs dist vs weather
    if do_plot_power_vs_dist_showing_weather:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=96)
        for irp in range(0, len(rp_list)):
            rp_list[irp].plot_power_vs_dist_with_weather(rp_list, rp_str_list, len(rp_list))
        plt.savefig(figurefile + '/route_power_vs_dist_weather.png')

    ##
    # plotting power vs dist ratios
    if do_plot_power_vs_dist_ratios:
        plot_power_vs_dist_ratios(rp_list, rp_str_list, 'power')

    if do_plot_fuel_vs_dist_ratios:
        plot_power_vs_dist_ratios(rp_list, rp_str_list, 'fuel')

    ##
    # write full fuel
    if do_write_fuel:
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
