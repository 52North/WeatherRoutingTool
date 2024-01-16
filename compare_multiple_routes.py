import os
import matplotlib.pyplot as plt

from WeatherRoutingTool.routeparams import RouteParams
import WeatherRoutingTool.utils.graphics as graphics

if __name__ == "__main__":
    routeDirPath = "C:/Users/Maneesha/Documents/GitHub/ROUTE/Test_Routes/"
    figureFilePath = "C:/Users/Maneesha/Documents/GitHub/ROUTE/Test_Routes/"

    plt.rcParams['font.size'] = 9
    fig, ax = plt.subplots(figsize=graphics.get_standard('fig_size'))
    ax.axis('off')
    ax.xaxis.set_tick_params(labelsize='large')
    fig, ax = graphics.generate_basemap(fig, depth=None, show_depth=False)

    rp_list = []
    legend_strings = []

    do_plot_min_route = True
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

    for irp in range(0, len(rp_list)):
        ax = rp_list[irp].plot_route(ax, graphics.get_colour(irp), legend_strings[irp])
    ax.legend()
    plt.savefig(figureFilePath + '/routes.png')

    if do_plot_min_route and min_file_name is not None:
        ax = min_rp.plot_route(ax, graphics.get_colour(irp+1), min_legend_string)
        ax.legend()
        plt.savefig(figureFilePath + '/routes_with_min_route.png')
