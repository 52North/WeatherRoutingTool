import argparse
import datetime
import math

import matplotlib.pyplot as plt

import WeatherRoutingTool.utils.graphics as graphics
from WeatherRoutingTool.config import Config
from WeatherRoutingTool.routeparams import RouteParams
from WeatherRoutingTool.utils.maps import Map
from WeatherRoutingTool.ship.ship import Tanker
from WeatherRoutingTool.weather_factory import WeatherFactory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Weather Routing Tool')
    parser.add_argument('-f', '--file', help="Config file name (absolute path)", required=True, type=str)

    args = parser.parse_args()
    if not args.file:
        raise RuntimeError("No config file name provided!")

    config = Config(file_name=args.file)
    config.print()

    windfile = config.WEATHER_DATA
    depthfile = config.DEPTH_DATA
    coursesfile = config.COURSES_FILE
    time_resolution = config.DELTA_TIME_FORECAST
    time_forecast = config.TIME_FORECAST
    departure_time = datetime.datetime.strptime(config.DEPARTURE_TIME, '%Y-%m-%dT%H:%MZ')
    lat1, lon1, lat2, lon2 = config.DEFAULT_MAP
    default_map = Map(lat1, lon1, lat2, lon2)

    wind_speed = 10
    u_comp = math.sin(45) * wind_speed
    v_comp = -math.cos(45) * wind_speed

    print('u_comp: ', u_comp)
    print('v_comp: ', v_comp)

    var_dict = {
        'thetao': 20,
        'Temperature_surface': 10,
        'Pressure_reduced_to_MSL_msl': 101325,
        'Pressure_surface': 101325,
        'u-component_of_wind_height_above_ground': u_comp,
        'v-component_of_wind_height_above_ground': v_comp,
        'VHM0': 1,
        'VMDR': 315
    }
    wf = WeatherFactory()
    wt = wf.get_weather(data_mode=config.DATA_MODE,
                        file_path=windfile,
                        departure_time=departure_time,
                        time_forecast=time_forecast,
                        time_resolution=time_resolution,
                        default_map=default_map,
                        var_dict=var_dict)
    fig, ax = plt.subplots(figsize=(12, 7))
    wt.plot_weather_map(fig, ax, departure_time, "wind")

    boat = Tanker(-99)
    boat.init_hydro_model_Route(windfile, coursesfile, depthfile)
    boat.set_boat_speed(6)

    lat, lon, time, sog = RouteParams.from_gzip_file(
        '/home/kdemmich/MariData/IMDC_paper/fixed_route_CB_pacific/HFData_voyage_16.gzip')

    waypoint_dict = RouteParams.get_per_waypoint_coords(lon, lat, time[0], sog.mean())

    ship_params = boat.get_fuel_per_time_netCDF(waypoint_dict['courses'], waypoint_dict['start_lats'],
                                                waypoint_dict['start_lons'], time)

    start = (lat[0], lon[0])
    finish = (lat[-1], lon[-1])

    rp = RouteParams(
        count=lat.shape[0],
        start=start,
        finish=finish,
        gcr=None,
        route_type='read_from_csv',
        time=waypoint_dict['travel_times'],
        lats_per_step=lat,
        lons_per_step=lon,
        azimuths_per_step=waypoint_dict['courses'],
        dists_per_step=waypoint_dict['dist'],
        starttime_per_step=time,
        ship_params_per_step=ship_params
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.remove()
    fig, ax = graphics.generate_basemap(fig=fig, depth=None,  start=start, finish=finish, show_depth=False)
    ax = rp.plot_route(ax, graphics.get_colour(0), rp)
    plt.show()

    ##
    # plotting power vs. distance
    fig, ax = plt.subplots(figsize=(12, 8), dpi=96)
    # rp_read1.plot_power_vs_dist(graphics.get_colour(0), rp_1_str)
    # rp_read2.plot_power_vs_dist(graphics.get_colour(1), rp_2_str)
    # rp_read3.plot_power_vs_dist(graphics.get_colour(2), rp_3_str)
    rp.plot_power_vs_dist(graphics.get_colour(3), '')
    plt.show()
