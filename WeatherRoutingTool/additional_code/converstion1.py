filepath='wind-router-master/wind-router-master/data/2019122212/20205150000_split13.grb'
#filepath='wind-router-master/wind-router-master/data/2019122212/2019122212f000'
#filepath='wind-router-master/updated1.grb'
import numpy as np

import pygrib as pg# Provides a high-level interface to  ECCODES C library for reading GRIB files

from scipy.interpolate import RegularGridInterpolator
"""GRIB is a file format for the storage and transport of gridded meteorological data,"""
import cfgrib
grib_data = cfgrib.open_datasets(filepath)
print('print grib data',grib_data)
#print(grib_data)
#u, _, _ = grib_data[1].data()  # U for Initial
#print(u)
grbs = pg.open(filepath)
print(grbs)
#u, _, _ = grbs[1].data()
u, _, _ = grbs[1].data()
v, _, _ = grbs[2].data()
print('printing u',u)
#plt.scatter(u,v)
#print('-------------------------------')
#plt.show()
#grbs = pg.open(filepath)
#u, a, b = grbs[1].data() # U for Initial
#print('length of u',len(u))
#print('length of v',len(v))

#print(len(a))
#print(b)


#print('hello')
#print('hello')

#v, _, _ = grbs[2].data() # V for Final
#lat1, lon1, lat2, lon2= [30, 0, 45, 40]
#u, lats_u, lons_u = grbs[1].data(lat1, lat2, lon1, lon2)
#v, lats_v, lo
# ns_v = grbs[2].data(lat1, lat2, lon1, lon2)




#print(len(u))
#print(u[0])
#print(lats_u)
#print(lons_u)

tws = np.sqrt(u * u + v * v)
twa = 180.0 / np.pi * np.arctan2(u, v) + 180.0#arctan:This mathematical function helps user to calculate inverse tangent for all x(being the array elements

#print(tws)
print('twa',twa)
#print(len(tws))
#print(len(twa))
lats_grid = np.linspace(-90, 90, 181)#Linespace : is a tool in Python for creating numeric sequences.
lons_grid = np.linspace(0, 360, 361)

print('latgrid',lats_grid)
print(lons_grid)
#np.file use to reserve the order of array
#.reshape(181,1) mens row 181 colum 1

f_twa = RegularGridInterpolator((lats_grid, lons_grid),np.flip(np.hstack((twa, twa[:, 0].reshape(181, 1))), axis=0),)
#hstack() function is used to stack the sequence of input arrays horizontally (i.e. column wise) to make a single array

f_tws = RegularGridInterpolator(
        (lats_grid, lons_grid),
        np.flip(np.hstack((tws, tws[:, 0].reshape(181, 1))), axis=0),
    )
print('f_tws',f_tws)
print('f_twa',f_twa)
