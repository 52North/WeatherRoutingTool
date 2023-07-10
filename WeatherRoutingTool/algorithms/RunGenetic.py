from DataUtils import *
from Genetic import *
from GeneticUtils import *

path = '/home/parichay/MariGeoRoute/Genetic/Data/CMEMS/'
data = loadData(path)
lon_min, lon_max, lat_min, lat_max = getBBox(-80, 32,-5, 47, data)
wave_height = data.isel(longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))
cost = cleanData(wave_height.data)
start, end = findStartAndEnd(40.7128,-74.0060,38.7223,-9.1393, wave_height)
#GeneticUtils.data = wave_height
set_data(wave_height, cost)
res = optimize(start, end, cost)

# get the best solution
best_idx = res.F.argmin()
best_x = res.X[best_idx]
best_f = res.F[best_idx]
route=best_x[0]

print(route)