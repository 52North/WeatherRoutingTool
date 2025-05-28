import argparse
import matplotlib.image as image
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from matplotlib.legend_handler import HandlerTuple

from astropy import units as u

import WeatherRoutingTool.utils.graphics as graphics
from WeatherRoutingTool.config import Config
from WeatherRoutingTool.utils.graphics import get_figure_path
from WeatherRoutingTool.utils.maps import Map
from WeatherRoutingTool.ship.ship import Tanker
from WeatherRoutingTool.weather_factory import WeatherFactory


def run_maripower_test_scenario(config_obj, calmfactor, windfactor, wavefactor, waypoint_dict, maripower_scenario,
                                weather_scenario):
    boat = Tanker(config_obj)
    boat.set_ship_property('WindForcesFactor', windfactor)
    boat.set_ship_property('WaveForcesFactor', wavefactor)
    boat.set_ship_property('CalmWaterFactor', calmfactor)

    print('Running scenario for ' + weather_scenario + ' with maripower setting ' + maripower_scenario)

    ship_params = boat.get_ship_parameters(waypoint_dict['courses'], waypoint_dict['start_lats'],
                                           waypoint_dict['start_lons'], waypoint_dict['time'], None, True)

    return ship_params


def plot_power_vs_courses(nominator_list, label_list, denominator_obj, courses,
                          figuredir, name, fuel_type, weather_type):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=96)
    ax.set_ylim(0.85, 1.2)
    colour = 0
    ratio = []

    for irp in range(0, len(nominator_list)):
        nominator = None
        denominator = None
        if fuel_type == 'power':
            nominator = nominator_list[irp].get_power()
            denominator = denominator_obj.get_power()
            ylabel = 'power consumption'
        else:
            nominator = nominator_list[irp].get_fuel()
            denominator = denominator_obj.get_fuel()
            ylabel = 'fuel consumption'

        mean_dev = (nominator/denominator).mean()
        plt.axhline(y=mean_dev, color=graphics.get_colour(colour), linestyle='dashed')
        if label_list[irp] == '':
            ratio_tmp, = ax.plot(courses, nominator/denominator, color=graphics.get_colour(colour),
                                 marker=graphics.get_marker(irp), linewidth=0, label=label_list[irp])
            colour = colour + 1
        else:
            ratio_tmp, = ax.plot(courses, nominator/denominator, color=graphics.get_colour(colour),
                                 marker=graphics.get_marker(irp), linewidth=0, label=label_list[irp])

        ratio.append(ratio_tmp)

    ship_tail_wind = image.imread('/home/kdemmich/MariData/IMDC_paper/ship_tail_wind.png')
    imagebox_tail_wind = OffsetImage(ship_tail_wind, zoom=0.25)
    ship_head_wind = image.imread('/home/kdemmich/MariData/IMDC_paper/ship_head_wind.png')
    imagebox_head_wind = OffsetImage(ship_head_wind, zoom=0.25)
    ab_left = AnnotationBbox(imagebox_tail_wind, (0, 1), (-300, -250), boxcoords="offset points", frameon=False)
    ab_mid = AnnotationBbox(imagebox_head_wind, (0, 1), (0, -250), boxcoords="offset points", frameon=False)
    ab_right = AnnotationBbox(imagebox_tail_wind, (0, 1), (+300, -250), boxcoords="offset points", frameon=False)

    ax.add_artist(ab_left)
    ax.add_artist(ab_mid)
    ax.add_artist(ab_right)

    ax.text(0.11, 0.74, 'dashed lines: averages', verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes)
    plt.xlabel('angular difference (degrees)')
    plt.ylabel(ylabel + ' modified/standard')
    plt.xticks()

    ax.legend(
        handles=[(ratio[1], ratio[0]), (ratio[3], ratio[2]), (ratio[5], ratio[4])],
        labels=[label_list[0], label_list[2], label_list[4]],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc='upper left', bbox_to_anchor=(0, 1), handlelength=0.6, frameon=False
    )
    ax.tick_params(top=True, right=True)
    fig.subplots_adjust(bottom=0.2)  # or whatever

    plt.savefig(os.path.join(figuredir, name + '.png'))

    plt.cla()
    plt.close()


def plot_polar_power(curve_list, label_list, courses, figuredir, name, fuel_type):

    fig, axes = plt.subplots(1, 1, dpi=200, subplot_kw={'projection': 'polar'}, figsize=(12, 15))
    courses_rad = np.radians(courses)

    for irp in range(0, len(curve_list)):
        curve = None
        if fuel_type == 'power':
            curve = curve_list[irp].get_power()
            ylabel = 'Leistung'
        else:
            curve = curve_list[irp].get_fuel()
            ylabel = 'Treibstoffverbrauch'
        plt.plot(courses_rad, curve, label=label_list[irp], color=graphics.get_colour(irp))

    axes.set_theta_direction(-1)
    axes.set_rlabel_position(-22.5)  
    axes.set_theta_zero_location("N")
    axes.grid(True)

    axes.set_title(ylabel, va='bottom')
    plt.legend(bbox_to_anchor=(0.5, -0.27), loc='lower center')
    plt.savefig(os.path.join(figuredir, name + '_polar.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Weather Routing Tool')
    parser.add_argument('-f', '--file', help="Config file name (absolute path)", required=True, type=str)
    parser.add_argument('-s', '--scenario', help="Specific scenario to run (e.g., 'original', '95perc_calm')", type=str)

    args = parser.parse_args()

    config = Config(file_name=args.file)
    config.print()

    routename = 'original_resistances_calm_weather'
    windfile = config.WEATHER_DATA
    figurepath = get_figure_path()
    time_resolution = config.DELTA_TIME_FORECAST
    time_forecast = config.TIME_FORECAST
    departure_time = datetime.strptime(config.DEPARTURE_TIME, '%Y-%m-%dT%H:%MZ')
    lat1, lon1, lat2, lon2 = config.DEFAULT_MAP
    default_map = Map(lat1, lon1, lat2, lon2)
    lat_start, lon_start, lat_end, lon_end = config.DEFAULT_ROUTE  
    coord_res = 0.1

    weather_type = 'calm_weather' 
    wind_speed = -99
    VHMO = -99
    VTPK = -99

    if weather_type == 'rough_weather':
        wind_speed = 12
        VHMO = 3.5
        VTPK = 9.0
    if weather_type == 'calm_weather':
        wind_speed = 2
        VHMO = 0.09
        VTPK = 1.5

    if wind_speed == -99 or VHMO == -99:
        raise ValueError('wind_speed or VHM0 not set!')

    u_comp = 0
    v_comp = - wind_speed

    var_dict = {
        'thetao': 20,
        'so': 33.5,
        'Temperature_surface': 293,
        'Pressure_reduced_to_MSL_msl': 101325,
        'u-component_of_wind_height_above_ground': u_comp,
        'v-component_of_wind_height_above_ground': v_comp,
        'VHM0': VHMO,
        'VMDR': 0,
        'VTPK': VTPK
    }

    maripower_test_scenarios_calm = {'original': 1., '95perc_calm': 0.95, '105perc_calm': 1.05, '80perc_wind': 1.,
                                     '120perc_wind': 1., '80perc_wave': 1., '120perc_wave': 1.}
    maripower_test_scenarios_wind = {'original': 1., '95perc_calm': 1., '105perc_calm': 1., '80perc_wind': 0.8,
                                     '120perc_wind': 1.2, '80perc_wave': 1., '120perc_wave': 1.}
    maripower_test_scenarios_wave = {'original': 1., '95perc_calm': 1., '105perc_calm': 1., '80perc_wind': 1.,
                                     '120perc_wind': 1., '80perc_wave': 0.8, '120perc_wave': 1.2}

    wt = WeatherFactory.get_weather(data_mode=config.DATA_MODE, file_path=windfile, departure_time=departure_time,
                                    time_forecast=time_forecast, time_resolution=time_resolution,
                                    default_map=default_map, var_dict=var_dict, coord_res=coord_res)

    courses = np.linspace(0, 360, 37) * u.degree       
    start_lats = np.repeat(lat_start, courses.shape)
    start_lons = np.repeat(lon_start, courses.shape)
    time = np.repeat(datetime.now(), courses.shape)
    travel_times = np.linspace(0, 36, 37)
    for i in range(0, time.shape[0]):
        time[i] = time[i] + timedelta(minutes=travel_times[i])

    waypoint_dict = {
        'courses': courses,
        'start_lats': start_lats,
        'start_lons': start_lons,
        'time': time
    }

    shipparams_vec = {}

    if args.scenario:
        key = args.scenario
        if key in maripower_test_scenarios_wind:
            shipparams_vec[key] = run_maripower_test_scenario(
                config,
                maripower_test_scenarios_calm[key],
                maripower_test_scenarios_wind[key],
                maripower_test_scenarios_wave[key],
                waypoint_dict,
                key, weather_type)
        else:
            print(f"Scenario '{key}' not found. Available scenarios: {', '.join(maripower_test_scenarios_wind.keys())}")
            exit()
    else:
        for key in maripower_test_scenarios_wind:
            shipparams_vec[key] = run_maripower_test_scenario(
                config,
                maripower_test_scenarios_calm[key],
                maripower_test_scenarios_wind[key],
                maripower_test_scenarios_wave[key],
                waypoint_dict,
                key, weather_type)


    plt.rcParams['font.size'] = graphics.get_standard('font_size')

    if not args.scenario or args.scenario in ['95perc_calm', '105perc_calm', '80perc_wind', '120perc_wind', '80perc_wave', '120perc_wave']:
        nominator_list = [
            shipparams_vec['95perc_calm'],
            shipparams_vec['105perc_calm'],
            shipparams_vec['80perc_wind'],
            shipparams_vec['120perc_wind'],
            shipparams_vec['80perc_wave'],
            shipparams_vec['120perc_wave']
        ]
        label_list = [
            r'$\pm5\,\%$ calm water resistance',
            '',
            r'$\pm20\,\%$ added resistance in wind',
            '',
            r'$\pm20\,\%$ added resistance in waves',
            ''
        ]
        plot_power_vs_courses(nominator_list, label_list, shipparams_vec['original'], courses, figurepath,
                              'inclusive_' + weather_type, 'power', weather_type)
