import argparse
import datetime as dt
import logging
import os

import matplotlib.pyplot as plt

import WeatherRoutingTool.utils.graphics as graphics
from WeatherRoutingTool.config import set_up_logging
from WeatherRoutingTool.constraints.constraints import *
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.utils.maps import Map
from WeatherRoutingTool.weather_factory import WeatherFactory


def plot_power_vs_dist(rp_list, rp_str_list, scenario_str, power_type='fuel'):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=96)
    for irp in range(0, len(rp_list)):
        rp_list[irp].plot_power_vs_dist(graphics.get_colour(irp), rp_str_list[irp], power_type, ax)

    ax.legend(loc='upper left', frameon=False)
    # ax.tick_params(top=True, right=True)
    ax.tick_params(labelleft=False, left=False, top=True)   # hide y labels
    ax.text(0.95, 0.96, scenario_str, verticalalignment='top', horizontalalignment='right',
            transform=ax.transAxes)
    plt.savefig(figurefile + '/' + power_type + '_vs_dist.png')


def plot_acc_power_vs_dist(rp_list, rp_str_list, power_type='fuel'):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=96)
    for irp in range(0, len(rp_list)):
        rp_list[irp].plot_acc_power_vs_dist(graphics.get_colour(irp), rp_str_list[irp], power_type)

    ax.legend(loc='upper center')
    # ax.set_ylim(0, 0.016)
    plt.savefig(figurefile + '/' + power_type + 'acc_vs_dist.png')


def plot_power_vs_coord(rp_list, rp_str_list, coordstring, power_type='fuel'):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=96)
    for irp in range(0, len(rp_list)):
        rp_list[irp].plot_power_vs_coord(ax, graphics.get_colour(irp), rp_str_list[irp], coordstring, power_type)
    ax.legend(loc='lower left')
    # ax.set_ylim(0, 0.016)
    plt.savefig(figurefile + '/' + power_type + '_vs_' + coordstring + '.png')


def plot_power_vs_dist_ratios(rp_list, rp_str_list, scenario_str, power_type='fuel'):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=96)
    ax.set_ylim(0.95, 1.08)
    colour = 0

    for irp in range(1, len(rp_list)):
        if rp_str_list[irp] == '':
            rp_list[irp].plot_power_vs_dist_ratios(rp_list[0], graphics.get_colour(colour),
                                                   rp_str_list[irp], power_type)
            colour = colour + 1
        else:
            rp_list[irp].plot_power_vs_dist_ratios(rp_list[0], graphics.get_colour(colour),
                                                   rp_str_list[irp], power_type)

    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), handlelength=0.1, frameon=False)
    ax.tick_params(top=True, right=True)
    ax.text(0.98, 0.96, scenario_str, verticalalignment='top', horizontalalignment='right',
            transform=ax.transAxes)

    ax.text(0.11, 0.76, 'dashed lines: averages', verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes)
    # plt.axhline(y=1, color='gainsboro', linestyle='-')
    plt.savefig(figurefile + '/' + power_type + '_vs_dist_ratios' + '.png')


if __name__ == "__main__":
    # Compare variations of resistances for specific routes

    parser = argparse.ArgumentParser(description='Weather Routing Tool')
    parser.add_argument('--base-dir', help="Base directory of route geojson files (absolute path)",
                        required=True, type=str)
    parser.add_argument('--figure-dir', help="Figure directory (absolute path)",
                        required=True, type=str)

    args = parser.parse_args()

    figurefile = args.figure_dir

    # Compare variations of resistances for specific routes
    '''filename1 = os.path.join(args.base_dir, "route_real_weather_original.json")
    filename2 = os.path.join(args.base_dir, "route_real_weather_95perc_calm.json")
    filename3 = os.path.join(args.base_dir, "route_real_weather_105perc_calm.json")
    filename4 = os.path.join(args.base_dir, "route_real_weather_80perc_wind.json")
    filename5 = os.path.join(args.base_dir, "route_real_weather_120perc_wind.json")
    filename6 = os.path.join(args.base_dir, "route_real_weather_80perc_wave.json")
    filename7 = os.path.join(args.base_dir, "route_real_weather_120perc_wave.json")


    rp_read1 = RouteParams.from_file(filename1)
    rp_read2 = RouteParams.from_file(filename2)
    rp_read3 = RouteParams.from_file(filename3)
    rp_read4 = RouteParams.from_file(filename4)
    rp_read5 = RouteParams.from_file(filename5)
    rp_read6 = RouteParams.from_file(filename6)
    rp_read7 = RouteParams.from_file(filename7)

    rp_1_str = 'Originalroute'
    rp_2_str = r'$\pm5\,\%$ calm water resistance'
    rp_3_str = ''
    rp_4_str = r'$\pm20\,\%$ added resistance in wind'
    rp_5_str = ''
    rp_6_str = r'$\pm20\,\%$ added resistance in waves'
    rp_7_str = ''

    scenario_str = 'scenario: Mediterranean Sea'

    rp_list = [rp_read1, rp_read2, rp_read3, rp_read4, rp_read5, rp_read6, rp_read7]
    rp_str_list = [rp_1_str, rp_2_str, rp_3_str, rp_4_str, rp_5_str, rp_6_str, rp_7_str]

    '''
    filename1 = "/home/kdemmich/MariData/IMDC_paper/Find_alternative_route_24_02_06/MedSea/Routes/route_real_weather_original.json"
    filename2 = "/home/kdemmich/MariData/IMDC_paper/Find_alternative_route_24_02_06/MedSea/Routes/min_time_route.json"

    rp_read1 = RouteParams.from_file(filename1)
    rp_read2 = RouteParams.from_file(filename2)

    rp_1_str = 'original route'
    rp_2_str = 'Isofuel routing'

    scenario_str = 'scenario: Mediterranean Sea'

    rp_list = [rp_read1, rp_read2]
    rp_str_list = [rp_1_str, rp_2_str]


    windfile = "/home/kdemmich/MariData/IMDC_paper/weather_imdc_route_16.nc"
    depth_data = ""
    set_up_logging()

    do_plot_weather = True
    do_plot_route = False
    do_plot_power_vs_dist = False
    do_plot_fuel_vs_dist = False
    do_plot_acc_fuel_vs_dist = False

    do_plot_power_vs_lon = False
    do_plot_fuel_vs_lon = False
    do_plot_power_vs_lat = False
    do_plot_fuel_vs_lat = False

    do_plot_power_vs_dist_showing_weather = False
    do_plot_power_vs_dist_ratios = False
    do_plot_fuel_vs_dist_ratios = False
    do_write_fuel = False

    ##
    # init weather
    departure_time = "2023-08-19T10:32Z"
    time_for_plotting = "2023-08-19T12:00Z"
    time_forecast = 60
    lat1, lon1, lat2, lon2 = (30, 10, 40, 35)

    #############################################################################
    plt.rcParams['font.size'] = graphics.get_standard('font_size')

    departure_time_dt = dt.datetime.strptime(departure_time, '%Y-%m-%dT%H:%MZ')
    plot_time = dt.datetime.strptime(time_for_plotting, '%Y-%m-%dT%H:%MZ')
    default_map = Map(lat1, lon1, lat2, lon2)

    if do_plot_weather:
        wt = WeatherFactory.get_weather("from_file", windfile, departure_time_dt, time_forecast, 3, default_map)

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.axis('off')
        ax.xaxis.set_tick_params(labelsize='large')
        fig, ax = graphics.generate_basemap(fig, None, rp_read1.start, rp_read1.finish, '', False)
        wt.plot_weather_map(fig, ax, plot_time, "wind")
        plt.show()

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
        plot_power_vs_dist(rp_list, rp_str_list, scenario_str, 'power')

    if do_plot_fuel_vs_dist:
        plot_power_vs_dist(rp_list, rp_str_list, scenario_str, 'fuel')

    ##
    # plotting  accumulated vs. distance

    if do_plot_acc_fuel_vs_dist:
        plot_acc_power_vs_dist(rp_list, rp_str_list, 'fuel')

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
        plot_power_vs_dist_ratios(rp_list, rp_str_list, scenario_str, 'power')

    if do_plot_fuel_vs_dist_ratios:
        plot_power_vs_dist_ratios(rp_list, rp_str_list, scenario_str, 'fuel')

    ##
    # write full fuel
    if do_write_fuel:
        print('Full fuel consumption:')
        print(rp_1_str + ': ' + str(rp_read1.get_full_fuel()))
        print(rp_2_str + ': ' + str(rp_read2.get_full_fuel()))
        # print(rp_3_str + ': ' + str(rp_read3.get_full_fuel()))
        # print(rp_4_str + ': ' + str(rp_read4.get_full_fuel()))

        print('Full travel dist:')
        print(rp_1_str + ': ' + str(rp_read1.get_full_dist()))
        print(rp_2_str + ': ' + str(rp_read2.get_full_dist()))
        # print(rp_3_str + ': ' + str(rp_read3.get_full_dist()))
        # print(rp_4_str + ': ' + str(rp_read4.get_full_dist()))

        print('Full travel time:')
        print(rp_1_str + ': ' + str(rp_read1.get_full_travel_time()))
        print(rp_2_str + ': ' + str(rp_read2.get_full_travel_time()))
        # print(rp_3_str + ': ' + str(rp_read3.get_full_travel_time('datetime')))
        # print(rp_4_str + ': ' + str(rp_read4.get_full_travel_time('datetime')))
