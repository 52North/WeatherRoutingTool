import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from astropy import units as u
from geovectorslib import geod
from matplotlib.figure import Figure
from PIL import Image

graphics_options = {'font_size': 20, 'fig_size': (12, 10)}


def get_standard(var):
    return graphics_options[var]


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


def create_maps(lat1, lon1, lat2, lon2, dpi, winds, n_maps):
    """Return map figure."""
    fig = Figure(figsize=(1600 / dpi, 800 * n_maps / dpi), dpi=dpi)
    fig.set_constrained_layout_pads(w_pad=4. / dpi, h_pad=4. / dpi)

    """Add gcrs between provided points to the map figure."""
    path = get_gcr_points(lat1, lon1, lat2, lon2, n_points=10)
    print(path)
    for i in range(n_maps):
        ax = fig.add_subplot(n_maps + 1, 1, i + 1, projection=ccrs.PlateCarree())
        ax.set_extent([lon1, lon2, lat1, lat2], crs=ccrs.PlateCarree())
        ax.add_feature(cf.LAND)
        ax.add_feature(cf.OCEAN)
        ax.add_feature(cf.COASTLINE)
        ax.gridlines(draw_labels=True)

        hour = i // 3 * 3
        u, v, lats, lons = winds[int(hour)]
        ax.barbs(lons, lats, u, v, length=5, sizes=dict(emptybarb=0.25, spacing=0.2, height=0.5), linewidth=0.95)

        lats = [x[0] for x in path]
        lons = [x[1] for x in path]
        ax = fig.get_axes()[0]
        ax.plot(lons, lats, 'r-', transform=ccrs.PlateCarree())
    return fig


def create_map(lat1, lon1, lat2, lon2, dpi):
    """Return map figure."""
    fig = Figure(figsize=(1200 / dpi, 420 / dpi), dpi=dpi)
    fig.set_constrained_layout_pads(w_pad=4. / dpi, h_pad=4. / dpi)
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0, top=1, wspace=0, hspace=0)
    # ax.set_extent([lon1, lon2, lat1, lat2], crs=ccrs.PlateCarree())
    ax.add_feature(cf.LAND)
    ax.add_feature(cf.OCEAN)
    ax.add_feature(cf.COASTLINE)
    ax.gridlines(draw_labels=True)
    return fig


def plot_barbs(fig, winds):
    """Add barbs to the map figure."""
    u = winds['u']
    v = winds['v']
    lats = winds['lats_u']
    lons = winds['lons_u']

    # rebinx=5  #CMEMS
    # rebiny=11

    # rebinx=1   #NCEP
    # rebiny=1

    rebinx = 10  # depth
    rebiny = 10

    u = rebin(u, rebinx, rebiny)
    v = rebin(v, rebinx, rebiny)
    lats = rebin(lats, rebinx, rebiny)
    lons = rebin(lons, rebinx, rebiny)

    ax = fig.get_axes()[0]
    ax.barbs(lons, lats, u, v, length=5, sizes=dict(emptybarb=0.25, spacing=0.2, height=0.5), linewidth=0.95)
    # ax.quiver(lons, lats, u, v)
    return fig


def plot_gcr(fig, lat1, lon1, lat2, lon2):
    """Add gcrs between provided points to the map figure."""
    path = get_gcr_points(lat1, lon1, lat2, lon2, n_points=10)
    lats = [x[0] for x in path]
    lons = [x[1] for x in path]

    ax = fig.get_axes()[0]
    ax.plot(lons, lats, 'g')
    return fig


'''
def plot_route(fig, route: RouteParams, colour):
    """
    Add isochrone to the map figure.
    Input: dictionary from move_boat_direct
    """
    ax = fig.get_axes()[0]
    lats = route.lats_per_step
    lons = route.lons_per_step
    ax = fig.get_axes()[0]
    # for i in range(len(lats)):
    #     ax.plot(lons[i], lats[i], 'ro')

    legend_entry = str(route.route_type) + ' (fuel: ' +  '%0.2f' % route.fuel + 't, time: ' + str(route.time) + 'h)'

    ax.plot(lons, lats, colour, label=legend_entry, transform=ccrs.PlateCarree())

    return fig
    '''


def plot_legend(fig):
    ax = fig.get_axes()[0]
    ax.legend()
    return fig


def get_colour(i):
    colours = ['darkred', 'gold', 'seagreen', 'peachpuff', 'darkviolet', 'crimson', 'olive',
               'skyblue', 'moccasin', 'plum', 'firebrick']
    if (i > 10):
        i = i-10
        print('Currently only 11 different colours available. Will use one that has already been used before: '
              'Colour=' + str(i))
    if (i > 20):
        print('Are you sure that you want to have so many curves in one plot?!')
    return colours[i]


def rebin(a, rebinx, rebiny):
    modx = a.shape[0] % rebinx
    mody = a.shape[1] % rebiny

    if not (modx == 0):
        for imod in range(0, modx):
            a = np.delete(a, 0, 0)

    if not (mody == 0):
        for imod in range(0, mody):
            a = np.delete(a, 0, 1)

    newshape_x = int(a.shape[0] / rebinx)
    newshape_y = int(a.shape[1] / rebiny)

    sh = newshape_x, rebinx, newshape_y, rebiny
    return a.reshape(sh).mean(-1).mean(1)


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
        contents = contents * u.kg/u.meter
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
            if not i == (bin_widths.shape[0]-1):
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


def generate_basemap(fig, depth, start=None, finish=None, title='', show_depth=True, show_gcr=False):
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    if show_depth:
        level_diff = 10
        cp = depth['depth'].plot.contourf(ax=ax, levels=np.arange(-100, 0, level_diff), transform=ccrs.PlateCarree())
        fig.colorbar(cp, ax=ax, shrink=0.7, label='Wassertiefe (m)', pad=0.1)

        fig.subplots_adjust(left=0.1, right=1.2, bottom=0, top=1, wspace=0, hspace=0)

    ax.add_feature(cf.LAND)
    ax.add_feature(cf.COASTLINE)
    ax.gridlines(draw_labels=True)
    # ax.set_extent((-1500000, 4000000, 3000000, 6000000), crs=ccrs.Mercator())

    if start is not None:
        ax.plot(start[1], start[0], marker="o", markerfacecolor="orange", markeredgecolor="orange", markersize=10)
        ax.plot(finish[1], finish[0], marker="o", markerfacecolor="orange", markeredgecolor="orange", markersize=10)

    if show_gcr:
        gcr = get_gcr_points(start[0], start[1], finish[0], finish[1], n_points=10)
        lats_gcr = [x[0] for x in gcr]
        lons_gcr = [x[1] for x in gcr]
        ax.plot(lons_gcr, lats_gcr, color="orange")

    plt.title(title)

    return fig, ax
