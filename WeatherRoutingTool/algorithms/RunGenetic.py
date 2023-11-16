from WeatherRoutingTool.algorithms.DataUtils import *
from WeatherRoutingTool.algorithms.Genetic import *
from WeatherRoutingTool.algorithms.GeneticUtils import *
import numpy as np
import matplotlib
from WeatherRoutingTool.routeparams import RouteParams
import WeatherRoutingTool.config as config
from datetime import datetime, timedelta



class RunGenetic():
    ncount : int
    count : int
    start : tuple
    finish : tuple
    departure_time: None
    figure_path : str
    weather_path: str
    fig: matplotlib.figure
    grc_azi : float
    route_ensemble : list
    route : np.array # temp
    pop_size : int
    n_offsprings : int 

    def __init__(self, start, finish, departure_time, weather_path, figure_path="") -> None:
        self.ncount = 20 #config.N_GEN
        self.count = 0
        self.start = start
        self.finish = finish
        self.departure_time = departure_time

        self.pop_size = 20 #config.POP_SIZE
        self.n_offsprings = 2 #config.N_OFFSPRINGS

        self.figure_path = figure_path
        self.weather_path = weather_path
        self.ship_params = None

        self.print_init()

    def execute_routing(self, boat, wt, constraint_list):

        data = loadData(self.weather_path)
        wave_height = data.VHM0.isel(time=0)
        genetic_util = GeneticUtils(departure_time=self.departure_time, boat=boat, wave_height=wave_height, constraint_list=constraint_list)
        start, end = findStartAndEnd(self.start[0], self.start[1], self.finish[0], self.finish[1], wave_height)
        res = optimize(start, end, self.pop_size, self.ncount, self.n_offsprings, genetic_util)
        # get the best solution
        best_idx = res.F.argmin()
        best_x = res.X[best_idx]
        best_f = res.F[best_idx]
        route=best_x[0]
        self.route = route
        _,self.ship_params = genetic_util.getPower([route], wave_height, self.departure_time)
        result = self.terminate(genetic_util)
        #print(route)
        #print(result)
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

    def terminate(self,genetic_util):
        form.print_line()
        print('Terminating...')

        lats, lons, route = genetic_util.index_to_coords(self.route)
        dists = distance(route)
        speed = 6
        diffs = time_diffs(speed, route)
        #ship_params = getPower()
        self.count = len(lats)

        dt = self.departure_time
        time = np.array([dt]*len(lats))
        times = np.array([t + timedelta(seconds=delta) for t, delta in zip(time, diffs)])

        route = RouteParams(

            count = self.count-3,
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
            ship_params_per_step = self.ship_params
        )

        self.check_destination()
        self.check_positive_power()
        return route

    def check_destination(self):
        pass

    def check_positive_power(self):
        pass
