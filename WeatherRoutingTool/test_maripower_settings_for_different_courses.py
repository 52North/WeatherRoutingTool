import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta

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
                                           waypoint_dict['start_lons'], waypoint_dict['time'], [], True)

    return ship_params


def plot_power_vs_courses(nominator_list, label_list, denominator_obj, courses, figuredir, name, fuel_type):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=96)
    ax.set_ylim(0.9, 1.1)

    for irp in range(0, len(nominator_list)):
        nominator = None
        denominator = None
        if fuel_type == 'power':
            nominator = nominator_list[irp].get_power()
            denominator = denominator_obj.get_power()
            ylabel = 'Leistung'
        else:
            nominator = nominator_list[irp].get_fuel()
            denominator = denominator_obj.get_fuel()
            ylabel = 'Treibstoffverbrauch'

        plt.plot(courses, nominator/denominator, marker='o', color=graphics.get_colour(irp), linewidth=0,
                 label=label_list[irp])
        plt.axhline(y=1, color='gray', linestyle='dashed')

    plt.xlabel('Course (degrees)')
    plt.ylabel(ylabel + ' Modifiziert/Standardwert')
    plt.xticks()

    ax.legend()

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
    axes.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    axes.set_theta_zero_location("N")
    axes.grid(True)

    axes.set_title(ylabel, va='bottom')
    plt.legend(bbox_to_anchor=(0.5, -0.27), loc='lower center')
    # plt.tight_layout()
    plt.savefig(os.path.join(figuredir, name + '_polar.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Weather Routing Tool')
    parser.add_argument('-f', '--file', help="Config file name (absolute path)", required=True, type=str)

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
    lat_start, lon_start, lat_end, lon_end = config.DEFAULT_ROUTE  # lat_end and lon_end are not used
    coord_res = 0.1

    weather_type = 'rough_weather'  # rough_weather, calm_weather
    wind_speed = -99
    VHMO = -99
    VTPK = -99

    if weather_type == 'rough_weather':
        wind_speed = 12.5
        VHMO = 3.5
        VTPK = 9.4
    if weather_type == 'calm_weather':
        wind_speed = 2.5
        VHMO = 0.1
        VTPK = 1.5

    if wind_speed == -99 or VHMO == -99:
        raise ValueError('wind_speed or VHM0 not set!')

    u_comp = 0
    v_comp = - wind_speed

    var_dict = {
        'thetao': 20,
        'so': 33.5,
        'Temperature_surface': 283,
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
    # fig, ax = plt.subplots(figsize=(12, 7))
    # wt.plot_weather_map(fig, ax, "2023-08-16T12:00:00", "wind")

    courses = np.linspace(0, 360, 37)  # scan courses from 0 to 360° in steps of 10°
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
    for key in maripower_test_scenarios_wind:
        shipparams_vec[key] = run_maripower_test_scenario(
            config,
            maripower_test_scenarios_calm[key],
            maripower_test_scenarios_wind[key],
            maripower_test_scenarios_wave[key],
            waypoint_dict,
            key, weather_type)

    plt.rcParams['font.size'] = graphics.get_standard('font_size')

    nominator_list = [shipparams_vec['95perc_calm'], shipparams_vec['105perc_calm']]
    label_list = ['95% Glattwasserwiderstand', '105% Glattwasserwiderstand']
    plot_power_vs_courses(nominator_list, label_list, shipparams_vec['original'], courses, figurepath,
                          'calmwaterres_' + weather_type, 'power')

    nominator_list = [shipparams_vec['80perc_wind'], shipparams_vec['120perc_wind']]
    label_list = ['80% Zusatzwiderstand Wind', '120% Zusatzwiderstand Wind']
    plot_power_vs_courses(nominator_list, label_list, shipparams_vec['original'], courses, figurepath,
                          'windres_' + weather_type, 'power')

    nominator_list = [shipparams_vec['80perc_wave'], shipparams_vec['120perc_wave']]
    label_list = ['80% Zusatzwiderstand Seegang', '120% Zusatzwiderstand Seegang']
    plot_power_vs_courses(nominator_list, label_list, shipparams_vec['original'], courses, figurepath,
                          'waveres_' + weather_type, 'power')

    curve_list = [shipparams_vec['original'], shipparams_vec['95perc_calm'], shipparams_vec['105perc_calm']]
    label_list = ['original', '95% Glattwasserwiderstand', '105% Glattwasserwiderstand']
    plot_polar_power(curve_list, label_list, courses, figurepath,
                     'calmwaterres_' + weather_type, 'power')

    curve_list = [shipparams_vec['original'], shipparams_vec['80perc_wind'], shipparams_vec['120perc_wind']]
    label_list = ['original', '80% Zusatzwiderstand Wind', '120% Zusatzwiderstand Wind']
    plot_polar_power(curve_list, label_list, courses, figurepath,
                     'windres_' + weather_type, 'power')

    curve_list = [shipparams_vec['original'], shipparams_vec['80perc_wave'], shipparams_vec['120perc_wave']]
    label_list = ['original', '80% Zusatzwiderstand Seegang', '120% Zusatzwiderstand Seegang']
    plot_polar_power(curve_list, label_list, courses, figurepath,
                     'waveres_' + weather_type, 'power')
