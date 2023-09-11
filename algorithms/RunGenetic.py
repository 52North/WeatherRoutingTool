from algorithms.DataUtils import *
from algorithms.Genetic import *
from algorithms.GeneticUtils import *
import numpy as np
import matplotlib
from routeparams import RouteParams
import config
from datetime import datetime, timedelta



class RunGenetic():
    ncount : int
    count : int
    start : tuple
    finish : tuple
    departure_time: np.ndarray
    figure_path : str
    fig: matplotlib.figure
    grc_azi : float
    route_ensemble : list
    route : np.array # temp



    pop_size : int
    n_offsprings : int

    def __init__(self, start, finish, departure_time, figure_path="") -> None:
        self.ncount = config.N_GEN
        self.count = 0
        self.start = start
        self.finish = finish
        self.departure_time = departure_time

        self.pop_size = config.POP_SIZE
        self.n_offsprings = config.N_OFFSPRINGS

        self.figure_path = figure_path

        self.print_init()

    def execute_routing(self):
        path = config.WEATHER_DATA
        data = loadData(path)
        lon_min, lon_max, lat_min, lat_max = getBBox(-80, 32, -5, 47, data)
        wave_height = data.VHM0.isel(time=0, longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))

        #wave_height = data.isel(longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))
        cost = cleanData(wave_height.data)
        start, end = findStartAndEnd(self.start[0], self.start[1], self.finish[0], self.finish[1], wave_height)
        set_data(wave_height, cost)
        res = optimize(start, end, cost, self.pop_size, self.ncount, self.n_offsprings)

        # get the best solution
        best_idx = res.F.argmin()
        best_x = res.X[best_idx]
        best_f = res.F[best_idx]
        route=best_x[0]
        self.route = route

        result = self.terminate()
        print(route)
        ship_params = getPower([route], wave_height)
        result.set_ship_params(ship_params)
        print(result)


        return result
    
    def print_init(self):
        print("Initializing Routing......")
        print('route from ' + str(self.start) + ' to ' + str(self.finish))
        #print('strart time ' + str(self.time))
        logger.info(form.get_log_step('route from ' + str(self.start) + ' to ' + str(self.finish),1))
        #logger.info(form.get_log_step('start time ' + str(self.time),1))

    def print_current_status(self):
        print("ALGORITHM SETTINGS:")
        print('start : ' + str(self.start))
        print('finish : ' + str(self.finish))
        print('generations: ' + str(self.ncount))
        print('pop_size: ' + str(self.pop_size))
        print('offsprings: ' + str(self.n_offsprings))

    def terminate(self):
        form.print_line()
        print('Terminating...')

        lats, lons, route = index_to_coords(self.route)
        dists = distance(route)
        speed = config.BOAT_SPEED
        diffs = time_diffs(speed, route)
        #ship_params = getPower()
        self.count = len(lats)

        dt = '2020.12.02 00:00:00' 
        dt_obj = datetime.strptime(dt, '%Y.%m.%d %H:%M:%S')
        time = np.array([dt_obj]*len(lats))
        times = np.array([t + timedelta(seconds=delta) for t, delta in zip(time, diffs)])
        #times = np.array([dt_obj]*len(lats))
        route = RouteParams(
            count = self.count,
            start = self.start,
            finish = self.finish,
            gcr = np.sum(dists),
            route_type = 'min_time_route',
            time = diffs, # time diffs
            lats_per_step = lats.to_numpy(),
            lons_per_step = lons.to_numpy(),
            azimuths_per_step = np.zeros(778),
            dists_per_step = dists, # dist of waypoints
            starttime_per_step = times, # time for each point
            ship_params_per_step = None
        )

        self.check_destination()
        self.check_positive_power()
        return route

    def check_destination(self):
        pass

    def check_positive_power(self):
        pass
