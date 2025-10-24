##
# Functionality to read json-files for different routes and visualise or compare them with varying focus.
##

import argparse
import datetime as dt

import matplotlib.pyplot as plt

from WeatherRoutingTool.config import set_up_logging
from WeatherRoutingTool.constraints.constraints import *
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.utils import graphics
from WeatherRoutingTool.utils.maps import Map
from WeatherRoutingTool.weather_factory import WeatherFactory


def plot_power_vs_dist(rp_list, rp_str_list, scenario_str, power_type='fuel'):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=96)
    ax.set_ylim(0, 5500)
    for irp in range(0, len(rp_list)):
        rp_list[irp].plot_power_vs_dist(graphics.get_colour(irp), rp_str_list[irp], power_type, ax)

    ax.legend(loc='upper left', frameon=False)
    ax.tick_params(top=True, right=True)
    # ax.tick_params(labelleft=False, left=False, top=True)   # hide y labels
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
    ax.set_ylim(0.7, 1.3)
    colour = 0

    for irp in range(1, len(rp_list)):
        if rp_str_list[irp] == '':
            rp_list[irp].plot_power_vs_dist_ratios(rp_list[0], graphics.get_colour(colour),
                                                   rp_str_list[irp], power_type)
            colour = colour + 1
        else:
            rp_list[irp].plot_power_vs_dist_ratios(rp_list[0], graphics.get_colour(colour),
                                                   rp_str_list[irp], power_type)
            colour = colour + 1

    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), handlelength=0.1, frameon=False)
    ax.tick_params(top=True, right=True)
    ax.text(0.8, 0.8, scenario_str, verticalalignment='top', horizontalalignment='right',
            transform=ax.transAxes)

    ax.text(0.2, 1.2, 'dashed lines: averages', verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes)
    # plt.axhline(y=1, color='gainsboro', linestyle='-')
    plt.savefig(figurefile + '/' + power_type + '_vs_dist_ratios' + '.png')


if __name__ == "__main__":
    # Compare variations of resistances for specific routes

    hist_dict = {
        'weather': False,
        'route': False,
        'power_vs_dist': False,
        'fuel_vs_dist': False,
        'acc_fuel_vs_dist': False,
        'power_vs_lon': False,
        'fuel_vs_lon': False,
        'power_vs_lat': False,
        'fuel_vs_lat': False,
        'power_vs_dist_showing_weather': False,
        'power_vs_dist_ratios': False,
        'fuel_vs_dist_ratios': False
    }

    parser = argparse.ArgumentParser(description='Weather Routing Tool')
    required_args = parser.add_argument_group('required arguments')
    optional_args = parser.add_argument_group('optional arguments')
    required_args.add_argument('--base-dir', help="Base directory of route geojson files (absolute path).",
                               required=True, type=str)
    required_args.add_argument('--figure-dir', help="Figure directory (absolute path).",
                               required=True, type=str)
    required_args.add_argument('--file-list', help="List of input route files (absolute paths).", nargs="*",
                               required=True, type=str)
    required_args.add_argument('--name-list',
                               help="List of legend entries for individual routes. "
                                    "Same ordering as for flag --file-list.",
                               nargs="*", required=True, type=str)
    required_args.add_argument('--hist-list',
                               help="List of histograms that shall be plotted. "
                                    "The following types are supported: "
                                    + str(hist_dict.keys()), nargs="*", required=True, type=str)
    optional_args.add_argument('--scenario-str',
                               help="String that can be added to the final plot for further description.",
                               required=False, default=' ', type=str)
    optional_args.add_argument('--wind-file', help="Absolute path to weather data.", required=False, default=' ',
                               type=str)

    # read arguments
    args = parser.parse_args()
    figurefile = args.figure_dir
    filelist = args.file_list
    rp_str_list = args.name_list
    hist_list = args.hist_list

    rp_list = []
    for path in filelist:
        rp_list.append(RouteParams.from_file(path))

    scenario_str = args.scenario_str
    windfile = args.wind_file
    depth_data = ""
    set_up_logging()

    # catch faulty configuration
    found_hist = False
    for option in hist_list:
        for hist in hist_dict:
            if option == hist:
                hist_dict[option] = True
                found_hist = True
        if not found_hist:
            parser.print_help()
            raise ValueError('The option "' + option + '" is not available for plotting!')
        found_hist = False

    if len(rp_list) != len(rp_str_list):
        parser.print_help()
        raise ValueError('Every histogram needs a name for the legend.')

    if (len(rp_list) < 2) and (hist_dict['power_vs_dist_ratios'] or hist_dict['fuel_vs_dist_ratios']):
        parser.print_help()
        raise ValueError('You need to pass at least two histograms for ratio plots.')

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

    if hist_dict['weather']:
        wt = WeatherFactory.get_weather("from_file", windfile, departure_time_dt, time_forecast, 3, default_map)

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.axis('off')
        ax.xaxis.set_tick_params(labelsize='large')
        fig, ax = graphics.generate_basemap(fig, None, rp_list[0].start, rp_list[0].finish, '', False)
        wt.plot_weather_map(fig, ax, plot_time, "wind")
        plt.show()

    ##
    # init Constraints
    # water_depth = WaterDepth('from_file', 20, default_map, depth_data)

    ##
    # plotting routes in depth profile
    if hist_dict['route']:
        fig, ax = plt.subplots(figsize=graphics.get_standard('fig_size'))
        ax.axis('off')
        ax.xaxis.set_tick_params(labelsize='large')
        fig, ax = graphics.generate_basemap(fig, None, rp_list[0].start, rp_list[0].finish, '', False)

        # ax = water_depth.plot_route_in_constraint(rp_read1, 0, fig, ax)
        for irp in range(0, len(rp_list)):
            ax = rp_list[irp].plot_route(ax, graphics.get_colour(irp), rp_str_list[irp])
        ax.legend()
        plt.savefig(figurefile + '/route_waterdepth.png')

    ##
    # plotting  vs. distance
    if hist_dict['power_vs_dist']:
        plot_power_vs_dist(rp_list, rp_str_list, scenario_str, 'power')

    if hist_dict['fuel_vs_dist']:
        plot_power_vs_dist(rp_list, rp_str_list, scenario_str, 'fuel')

    ##
    # plotting  accumulated vs. distance

    if hist_dict['acc_fuel_vs_dist']:
        plot_acc_power_vs_dist(rp_list, rp_str_list, 'fuel')

    ##
    # plotting power vs. coordinate
    if hist_dict['power_vs_lat']:
        plot_power_vs_coord(rp_list, rp_str_list, 'lat', 'power')

    if hist_dict['fuel_vs_lat']:
        plot_power_vs_coord(rp_list, rp_str_list, 'lat', 'fuel')

    if hist_dict['power_vs_lon']:
        plot_power_vs_coord(rp_list, rp_str_list, 'lon', 'power')

    if hist_dict['fuel_vs_lon']:
        plot_power_vs_coord(rp_list, rp_str_list, 'lon', 'fuel')

    ##
    # plotting power vs dist vs weather
    if hist_dict["power_vs_dist_showing_weather"]:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=96)
        for irp in range(0, len(rp_list)):
            rp_list[irp].plot_power_vs_dist_with_weather(rp_list, rp_str_list, len(rp_list))
        plt.savefig(figurefile + '/route_power_vs_dist_weather.png')

    ##
    # plotting power vs dist ratios
    if hist_dict["power_vs_dist_ratios"]:
        plot_power_vs_dist_ratios(rp_list, rp_str_list, scenario_str, 'power')

    if hist_dict["fuel_vs_dist_ratios"]:
        plot_power_vs_dist_ratios(rp_list, rp_str_list, scenario_str, 'fuel')

    ##
    # write fuel consumption, travel distance and time
    print('Full fuel consumption:')
    for irp in range(0, len(rp_list)):
        print(rp_str_list[irp] + ': ' + str(rp_list[irp].get_full_fuel()))

    print('Full travel dist:')
    for irp in range(0, len(rp_list)):
        print(rp_str_list[irp] + ': ' + str(rp_list[irp].get_full_dist()))

    print('Full travel time:')
    for irp in range(0, len(rp_list)):
        print(rp_str_list[irp] + ': ' + str(rp_list[irp].get_full_travel_time()))
