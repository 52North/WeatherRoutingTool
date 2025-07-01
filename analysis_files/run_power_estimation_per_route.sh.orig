#!/bin/bash

#####################################################
# paths that need to be set
working_dir=/home/<user>/<wrt_work_dir>
config_file=/home/<user>/<path_to_config_json>
gzip_route=/home/<user>/<path_to_gzip_data>
virtual_env=venv/bin/activate
routes_path=/home/<user>/<path_to_work_dir>/Routes
figure_dir=/home/<user>/<path_to_work_dir>/Figures

# paths to routes that shall be visualised
datapath=${routes_path}/route_data.json
maripower=${routes_path}/route_original_maripower.json
maripower_only_wind=${routes_path}/route_maripower_only_wind.json
dpm=${routes_path}/route_dpm.json


#####################################################
# read argument, store it in variable option and reset it afterwards
option=$1
shift

# exit with error if any error occurs during the individual function calls
set -e

#load environment
cd ${working_dir}/WeatherRoutingTool
source ${virtual_env}
source load_wrt.sh

#####################################################
# analyse fuel consumption using maripower for different settings
function maripower()
{
	# get fuel estimation for full maripower
	name_original='original_maripower'
	calm_original=1.
	wave_original=1.
	wind_original=1.  

	python3 ${working_dir}/WeatherRoutingTool/WeatherRoutingTool/test_maripower_settings_for_specific_route.py -f ${config_file} -r ${gzip_route} --write-geojson=True -wave=${wave_original} -wind=${wind_original} -calm=${calm_original} -n=${name_original} -bt='maripower'

	# get fuel estimation for maripower without wind
	name_wout_wind='maripower_no_wind'
	calm_wout_wind=1.
	wave_wout_wind=1.
	wind_wout_wind=0.  

	python3 ${working_dir}/WeatherRoutingTool/WeatherRoutingTool/test_maripower_settings_for_specific_route.py -f ${config_file} -r ${gzip_route} --write-geojson=True -wave=${wave_wout_wind} -wind=${wind_wout_wind} -calm=${calm_wout_wind} -n=${name_wout_wind} -bt='maripower'

	# get fuel estimation for maripower without wind
	name_only_wind='maripower_only_wind'
	calm_only_wind=1.
	wave_only_wind=0.
	wind_only_wind=1.  

	python3 ${working_dir}/WeatherRoutingTool/WeatherRoutingTool/test_maripower_settings_for_specific_route.py -f ${config_file} -r ${gzip_route} --write-geojson=True -wave=${wave_only_wind} -wind=${wind_only_wind} -calm=${calm_only_wind} -n=${name_only_wind} -bt='maripower'
}

# analyse fuel consumption using direct power method
function dpm()
{

	# get fuel estimation for dpm
	name_dpm='dpm'

	python3 ${working_dir}/WeatherRoutingTool/WeatherRoutingTool/test_maripower_settings_for_specific_route.py -f  -r ${gzip_route} --write-geojson=True -n=${name_dpm} 
}

# analyse fuel consumption from data
function data()
{
	# get fuel estimation for maripower without wind
	name_only_wind='data'

	python3 ${working_dir}/WeatherRoutingTool/WeatherRoutingTool/test_maripower_settings_for_specific_route.py -f ${config_file} -r ${gzip_route} --write-geojson=True  -n=${name_only_wind} -bt=data
}

# plot routes that have been generated above
function graphics()
{
  	python3 ${working_dir}/WeatherRoutingTool/compare_routes.py --base-dir ${routes_path} --figure-dir ${figure_dir} --file-list ${datapath}  --name-list 'data' 'maripower' 'direct power method' --hist-list 'power_vs_dist_ratios' 'fuel_vs_dist_ratios' 'power_vs_dist' 'fuel_vs_dist'
}

# help functionality
function analysis_help()
{
	python3 ${working_dir}/WeatherRoutingTool/WeatherRoutingTool/test_maripower_settings_for_specific_route.py -h
}

function graphics_help()
{
	python3 ${working_dir}/WeatherRoutingTool/compare_routes.py -h
}

# option handling
echo ${option}
case ${option} in 
	maripower)
	  maripower
	;;
	data)
	  data
	;;
	dpm)
	  dpm
	;;
	graphics)
	  graphics
	;;
	analysis_help)
	  analysis_help
	;;
	graphics_help)
	  graphics_help
	;;
	*)
	  maripower
	  data
	  dpm
	  graphics
	;;
esac
