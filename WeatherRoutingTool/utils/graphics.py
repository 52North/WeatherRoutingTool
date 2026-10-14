import csv
import datetime

import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from astropy import units as u
from geovectorslib import geod
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PIL import Image

graphics_options = {'font_size': 20, 'fig_size': (12, 10)}


def get_standard(var):
    return graphics_options[var]


def get_standard_fig():
    fig = Figure(figsize=get_standard('fig_size'), dpi=100)
    matplotlib.rcParams.update({'font.size': get_standard('font_size')})
    return fig


def get_gcr_points(lat1, lon1, lat2, lon2, n_points=10):
    """Discretize gcr between two scalar coordinate points."""
    points = [(lat1, lon1)]

    inv = geod.inverse([lat1], [lon1], [lat2], [lon2])
    dist = inv['s12'] / (n_points)

    for i in range(n_points):
        dir = geod.direct(lat1, lon1, inv['azi1'], dist)
        points.append((dir['lat2'][0], dir['lon2'][0]))
        lat1 = dir['lat2'][0]
        lon1 = dir['lon2'][0]
        inv = geod.inverse([lat1], [lon1], [lat2], [lon2])

    return points


def plot_gcr(fig, lat1, lon1, lat2, lon2):
    """Add gcrs between provided points to the map figure."""
    path = get_gcr_points(lat1, lon1, lat2, lon2, n_points=10)
    lats = [x[0] for x in path]
    lons = [x[1] for x in path]

    ax = fig.get_axes()[0]
    ax.plot(lons, lats, 'g')
    return fig


def get_colour(i):
    # colorblind friendly (mixture of tableau-colorblind10 & seaborn colorblind)
    colours = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442',
               '#56B4E9', '#006BA4', '#ABABAB', '#595959', '#FFBC79']
    if i > 18:
        print('Are you sure that you want to have so many curves in one plot?!')
        i = i - 18
    if i > 9:
        i = i - 9
        print('Currently only 11 different colours available. Will use one that has already been used before: '
              'Colour=' + str(i))
    return colours[i]


def get_str_color(i):
    # colors optimised for visibility on map with weather
    colours = ['#0072B2', '#D55E00', '#009E73', '#FFBC79']

    return colours[i]


def get_marker(i):
    markers = ['o', 's', 'd', 'P', 'D', 'x', 'p']
    if i > 6:
        i = i - 7
        print('Currently only 5 different colours available. Will use one that has already been used before: '
              'Colour=' + str(i))
    return markers[i]


def get_linestyle(i):
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']

    if i > 3:
        i = i - 4
        print('Currently only 4 different linestyles are available. Will use one that has already been used before: '
              'Linestyle=' + str(i))
    print('using linestyle: ', linestyles[i])
    return linestyles[i]


def merge_figs(path, ncounts):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=500)
    ax.axis('off')
    image_list = []
    impath = ''

    fig.subplots_adjust(left=0.01, right=0.98, bottom=0, top=1, wspace=0, hspace=0)

    for iIm in range(1, ncounts + 1):
        impath = path + 'fig' + str(iIm) + 'p.png'
        print('Reading image ', impath)
        im_temp = Image.open(impath)
        im_plot = plt.imshow(im_temp)
        image_list.append([im_plot])

    ani = animation.ArtistAnimation(fig, image_list, interval=1000, blit=True, repeat_delay=10)

    writergif = animation.PillowWriter(fps=1, bitrate=1000)
    ani.save(path + str(ani) + '.gif', writer=writergif)


def get_figure_path():
    figure_path = os.getenv('WRT_FIGURE_PATH')
    if figure_path:
        # Set figure path to None if the provided string is not a directory or not writable
        if not os.path.isdir(figure_path) or not os.access(figure_path, os.W_OK):
            figure_path = None
    return figure_path


def get_hist_values_from_boundaries(bin_boundaries, contend_unnormalised):
    centres = np.array([])
    widths = np.array([])
    contents = np.array([])
    for i in range(0, bin_boundaries.shape[0] - 1):
        width_temp = (bin_boundaries[i + 1] - bin_boundaries[i])
        cent_temp = bin_boundaries[i] + width_temp / 2
        cont_temp = contend_unnormalised[i] / width_temp
        centres = np.append(centres, cent_temp)
        widths = np.append(widths, width_temp)
        contents = np.append(contents, cont_temp)
    return {"bin_content": contents, "bin_centres": centres, "bin_widths": widths}


def get_hist_values_from_widths(bin_widths, contend_unnormalised, power_type):
    centres = np.array([]) * u.meter
    contents = np.array([])
    if power_type == 'fuel':
        contents = contents * u.kg / u.meter
    elif power_type == 'speed':
        contents = contents * u.meter / u.second
    elif power_type == 'relative_power':
        contents = contents
    else:
        contents = contents * u.Watt
    cent_temp = 0 * u.meter
    debug = False

    # ToDo: use logger.debug and args.debug
    if debug:
        print('bin_width:', bin_widths)
        print('contend_unnormalised:', contend_unnormalised)
    for i in range(0, bin_widths.shape[0]):
        cont_temp = 0
        cent_temp = cent_temp + bin_widths[i] / 2
        if (bin_widths[i] > 0) or (contend_unnormalised[i] != -99):
            cont_temp = contend_unnormalised[i]
            if power_type == 'fuel':
                cont_temp = cont_temp / bin_widths[i]
            centres = np.append(centres, cent_temp)
            contents = np.append(contents, cont_temp)
            cent_temp = cent_temp + bin_widths[i] / 2
        else:
            if not i == (bin_widths.shape[0] - 1):
                raise ValueError('Unexpected behaviour!')
            bin_widths = bin_widths[:-1]

    if debug:
        print('shape bin_contents: ', contents.shape)
        print('shape bin_centres: ', centres.shape)
        print('shape bin_widths: ', bin_widths.shape)
    return {"bin_contents": contents, "bin_centres": centres, "bin_widths": bin_widths}


def get_accumulated_dist(dist_arr):
    dist_acc = np.array([])
    full_dist = 0

    for dist in dist_arr:
        dist_acc_temp = dist + full_dist
        dist_acc = np.append(dist_acc, dist_acc_temp)
        full_dist = full_dist + dist

    return dist_acc


def set_graphics_standards(ax):
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)
    plt.rcParams['font.size'] = '20'
    return ax


def generate_basemap(
        map_coords,
        depth,
        start=None,
        finish=None,
        title='',
        show_depth=True,
        show_gcr=False,
        wt=None,
        wt_timestamp=None
):
    plt.rcParams['font.size'] = get_standard('font_size')
    (min_lat, max_lat, min_lon, max_lon) = map_coords

    fig = plt.figure(figsize=get_standard('fig_size'))
    fig_width, fig_height = get_standard('fig_size')

    input_crs = ccrs.PlateCarree()
    output_crs = ccrs.Mercator()
    ax = fig.add_subplot(1, 1, 1, projection=output_crs)

    # coordinate extension in Mercator crs
    x_min, y_min = output_crs.transform_point(min_lon, min_lat, input_crs)  # lower left corner
    x_max, y_max = output_crs.transform_point(max_lon, max_lat, input_crs)  # upper right corner
    delta_x = x_max - x_min
    delta_y = y_max - y_min

    projected_ratio = delta_x / delta_y
    target_ratio = fig_width / fig_height

    x_min_new, x_max_new = x_min, x_max
    y_min_new, y_max_new = y_min, y_max
    if projected_ratio < target_ratio:
        new_delta_x = delta_y * target_ratio
        center_x = (x_min + x_max) / 2

        x_min_new = center_x - new_delta_x / 2
        x_max_new = center_x + new_delta_x / 2
    elif projected_ratio > target_ratio:
        new_delta_y = delta_x / target_ratio
        center_y = (y_min + y_max) / 2

        y_min_new = center_y - new_delta_y / 2
        y_max_new = center_y + new_delta_y / 2

    ax.set_global()

    if show_depth:
        level_diff = 10
        depth['z'].plot.contourf(ax=ax, levels=np.arange(-100, 0, level_diff), transform=ccrs.PlateCarree())
        # fig.colorbar(cp, ax=ax, shrink=0.7, label='Wassertiefe (m)', pad=0.1)

        fig.subplots_adjust(left=0.1, right=1.2, bottom=0, top=1, wspace=0, hspace=0)

    if wt is not None:
        time = datetime.datetime.strptime(wt_timestamp, '%Y-%m-%dT%H:%MZ')

        # plot windspeed
        u = wt.ds['u-component_of_wind_height_above_ground'].sel(
            height_above_ground=10,
            latitude=slice(min_lat, max_lat),
            longitude=slice(min_lon, max_lon),
        )
        u = u.sel(time=time, method='nearest')
        v = wt.ds['v-component_of_wind_height_above_ground'].sel(
            height_above_ground=10,
            latitude=slice(min_lat, max_lat),
            longitude=slice(min_lon, max_lon),
        )
        v = v.sel(time=time, method='nearest')
        windspeed = np.sqrt(u ** 2 + v ** 2)

        level_diff = 0.1
        cfplot = windspeed.plot.contourf(
            ax=ax,
            levels=np.arange(2, 7, level_diff),
            transform=ccrs.PlateCarree(),
            zorder=0,
            alpha=0.7
        )
        cbar = cfplot.colorbar
        cbar.set_label('wind speed (m/s)', fontsize=20, labelpad=10)

        # plot wind barbs
        stride = 10
        lons = wt.ds["longitude"].sel(longitude=slice(min_lon, max_lon)).values[::stride]
        lats = wt.ds["latitude"].sel(latitude=slice(min_lat, max_lat)).values[::stride]
        u_wind = u.values[::stride, ::stride]
        v_wind = v.values[::stride, ::stride]
        lon_2d, lat_2d = np.meshgrid(lons, lats)

        ax.barbs(
            lon_2d, lat_2d,
            u_wind, v_wind,
            transform=ccrs.PlateCarree(),
            length=5,
            linewidth=0.8,
            color='black',
            zorder=0.5
        )

    if start is not None:
        ax.plot(start[1], start[0], marker="o", markerfacecolor="orange", markeredgecolor="orange", markersize=10,
                transform=input_crs)
        ax.plot(finish[1], finish[0], marker="o", markerfacecolor="orange", markeredgecolor="orange", markersize=10,
                transform=input_crs)

    if show_gcr:
        gcr = get_gcr_points(start[0], start[1], finish[0], finish[1], n_points=10)
        lats_gcr = [x[0] for x in gcr]
        lons_gcr = [x[1] for x in gcr]
        ax.plot(lons_gcr, lats_gcr, color="orange")

    ax.add_feature(cf.LAND, zorder=1)
    ax.add_feature(cf.COASTLINE)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.right_labels = False
    ax.set_extent([x_min_new, x_max_new, y_min_new, y_max_new], crs=output_crs)

    fig.tight_layout()
    plt.title(title)
    return fig, ax


def write_graph_to_csv(path, x, y):
    with open(path, 'w', newline='') as csvfile:
        graphwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for i in range(0, len(x)):
            graphwriter.writerow([x[i], y[i]])


def get_route_lc(X):
    lats = X[:, 0]
    lons = X[:, 1]
    speed = X[:, 2]

    points = np.array([lons, lats]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = Normalize(vmin=0, vmax=20)
    lc = LineCollection(segments, cmap='viridis', norm=norm, transform=ccrs.Geodetic())
    lc.set_array(speed)
    lc.set_linewidth(1)
    return lc
