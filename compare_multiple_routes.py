import os
import matplotlib.pyplot as plt
import matplotlib.colors as color
import matplotlib.collections as collections
import numpy as np
import xarray as xr
import matplotlib as mpl

from WeatherRoutingTool.routeparams import RouteParams
import WeatherRoutingTool.utils.graphics as graphics


def plot_status(ax, routparam, cmap, norm, x_min, x_max, y_min, y_max, legend_strings, irp):
    y = routparam.lats_per_step
    x = routparam.lons_per_step

    if x_min >= x.min():
        x_min = x.min()
    if x_max <= x.max():
        x_max = x.max()
    if y_min >= y.min():
        y_min = y.min()
    if y_max <= y.max():
        y_max = y.max()

    status_val = routparam.ship_params_per_step.get_status()
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = collections.LineCollection(segments, linewidths=6)
    lc.set_array(np.asarray(status_val))
    lc.set_cmap(cmap)
    lc.set_norm(norm)
    ax.add_collection(lc)
    ax = routparam.plot_route(ax, graphics.get_colour(irp), legend_strings, True)
    return ax, x_min, x_max, y_min, y_max


if __name__ == "__main__":
    routeDirPath = "/home/kdemmich/MariData/Debug_Multiple_Routes/Speedy_Isobased/MedSea/Routes"
    figureFilePath = "/home/kdemmich/MariData/Debug_Multiple_Routes/Speedy_Isobased/MedSea/Figures"

    plt.rcParams['font.size'] = 9
    fig, ax = plt.subplots(figsize=graphics.get_standard('fig_size'))
    ax.axis('off')
    ax.xaxis.set_tick_params(labelsize='large')
    fig, ax = graphics.generate_basemap(fig, depth=None, show_depth=False)

    rp_list = []
    legend_strings = []

    do_plot_min_route = True
    do_plot_status = True
    min_file_name = None

    for file in os.listdir(routeDirPath):
        if file.endswith(".json"):
            route_file = os.path.join(routeDirPath, file)
            rp = RouteParams.from_file(route_file)
            rp_full_fuel = rp.get_full_fuel()
            if file.startswith('min_time_route.json'):
                min_file_name = file
                min_rp = rp
                min_legend_string = file+' '+"{:.4f}".format(rp_full_fuel) + 't'
                continue
            else:
                rp_list.append(rp)
                legend_strings.append(file+' '+"{:.4f}".format(rp_full_fuel) + 't')
                continue
        else:
            continue

    if len(rp_list) == 0:
        irp = 0
    else:
        for irp in range(0, len(rp_list)):
            ax = rp_list[irp].plot_route(ax, graphics.get_colour(irp), legend_strings[irp])
        ax.legend()
        plt.savefig(figureFilePath + '/routes.png')

    if do_plot_min_route and min_file_name is not None:

        ax = min_rp.plot_route(ax, graphics.get_colour(irp+1), min_legend_string)
        ax.legend()
        plt.savefig(figureFilePath + '/routes_with_min_route.png')

    if do_plot_status:
        x_min = 180
        x_max = -180
        y_min = 90
        y_max = -90

        fig, ax = plt.subplots(figsize=graphics.get_standard('fig_size'))
        ax.axis('off')
        ax.xaxis.set_tick_params(labelsize='large')
        fig, ax = graphics.generate_basemap(fig, depth=None, show_depth=False)

        cmap = (color.ListedColormap(['green', 'yellow', 'red']))
        bounds = [0, 1.1, 2.2, 3.1]
        norm = color.BoundaryNorm(bounds, cmap.N)
        for irp in range(0, len(rp_list)):
            ax, x_min, x_max, y_min, y_max = plot_status(ax, rp_list[irp], cmap,
                                                         norm, x_min, x_max, y_min, y_max, legend_strings[irp], irp)

        if do_plot_min_route and min_file_name is not None:
            ax, x_min, x_max, y_min, y_max = plot_status(ax, min_rp, cmap, norm,
                                                         x_min, x_max, y_min, y_max, min_legend_string, irp+1)

        ax.legend()

        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                            location='right', shrink=0.6, pad=0.15)
        cbar.ax.get_yaxis().set_ticks([])
        for j, lab in enumerate(['$OK$', '$Warning$', '$Error$']):
            cbar.ax.text(2.1, j * 1.1/0.8, lab, ha='left', va='center')
        cbar.ax.set_title('Status code')

        plt.xlim(x_min - 2, x_max + 2)
        plt.ylim(y_min - 2, y_max + 2)
        plt.savefig(figureFilePath + '/routes_with_status.png')
