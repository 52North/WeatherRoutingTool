import argparse
import os

import WeatherRoutingTool.utils.graphics as graphics
import matplotlib.pyplot as plt
import pandas as pd


def compare_ga_convergence(paths, names, figuredir):
    graph_pd = None
    for i_path in range(0, len(paths)):
        pd_temp = pd.read_csv(filepath_or_buffer=paths[i_path], sep=" ", names=["n_gen", names[i_path]])
        pd_temp = pd_temp.set_index('n_gen')
        if graph_pd is None:
            graph_pd = pd_temp
        else:
            graph_pd = graph_pd.join(pd_temp, on="n_gen")

    graphics.get_standard_fig()
    graph_pd.plot()
    plt.show()
    # plt.savefig(os.path.join(figuredir, 'convergence.png'))


if __name__ == "__main__":

    hist_dict = {
        'convergence': False,
        'running_metric': False,
    }

    parser = argparse.ArgumentParser(description='Weather Routing Tool')
    required_args = parser.add_argument_group('required arguments')
    # optional_args = parser.add_argument_group('optional arguments')
    required_args.add_argument('--csv-list', help="Base directory of route geojson files (absolute path).",
                               nargs="*", required=True, type=str)
    required_args.add_argument('--figure-dir', help="Figure directory (absolute path).",
                               required=True, type=str)
    required_args.add_argument('--hist-list',
                               help="List of histograms that shall be plotted. "
                                    "The following types are supported: "
                                    + str(hist_dict.keys()), nargs="*", required=True, type=str)
    required_args.add_argument('--name-list',
                               help="List of legend entries for individual routes. "
                                    "Same ordering as for flag --file-list.",
                               nargs="*", required=True, type=str)

    # read arguments
    args = parser.parse_args()
    figurefile = args.figure_dir
    csv_list = args.csv_list
    hist_list = args.hist_list
    name_list = args.name_list

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

    if len(csv_list) != len(name_list):
        parser.print_help()
        raise ValueError('Every histogram needs a name for the legend.')

    if hist_dict["convergence"]:
        compare_ga_convergence(csv_list, name_list, figurefile)
