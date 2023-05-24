"""Routing functions."""
from geovectorslib import geod
from global_land_mask import globe
#global-land-mask, contains scripts for checking whether a lat/lon point is on land or sea
import numpy as np
import datetime as dt
from RoutingAlg import Isochrone
from polars import boat_speed_function
from weather import wind_function
from scipy.stats import binned_statistic


def move_boat_direct(lats, lons, hdgs, boat, winds,start_time, delta_time, verbose=False):
    print('boat speed 1')
    winds = wind_function(winds, (lats, lons), start_time)
    print('printin winds from move boat',hdgs)
    print('boat speed 2')
    twa = winds['twa']
    tws = winds['tws']
    print('printing lats from router.py',lats)
    print('printing lons from router.py',lons)
    wind = {'tws': tws, 'twa': twa - hdgs}

    bs = boat_speed_function(boat, wind)

    if verbose:
        print('TWA: ', twa)
        print('TWS: ', tws)
        print('HDG: ', hdgs)
        print('WA:  ', twa - hdgs)
        print('BS:  ', bs)

    # distance in meters
    dist = delta_time * bs
    print('distance',dist)
    print('printing lons above gcr',lons )
    # move boat
    gcr = geod.direct(lats, lons, hdgs, dist)
    #print('gcr lats',gcr['lats'])
    print('printing gcr',gcr['iterations'])
    #gcr=geod.inverse(lats,lons,hdgs,dist)
    print('gcr',gcr)


    return {'azi1': hdgs,
            's12': dist,
            'lats2': gcr['lat2'],
            'lons2': gcr['lon2'],
            't12': delta_time}

## initial isochrone
def prune_isochrone(iso: Isochrone, x, y, bins, trim=True):
    """
    Binned statistic.
    +            iso2 = prune_isochrone(iso2, 'azi02', 's02', bins, True)
    print('iso2 ',iso2)  #

            Parameters:
                iso: isochrone dictionary
                x: values to binarize
                y: values to apply max to
                bins: bins edges, dimension is n_bins + 1
                trim: whether return just one of max values
            Returns:
                pruned isochrone dictionary with max values in each bin
    """
    idxs = []
    print('empty iso',iso)
    arr_x = getattr(iso, x)
    print(arr_x,'arrx')
    arr_y = getattr(iso, y)
    print(arr_y, 'arry')

    bin_stat, bin_edges, bin_number = binned_statistic(
        arr_x, arr_y, statistic=np.nanmax, bins=bins)
    print(binned_statistic,'binned stat')
    print('bin_edgess',bin_edges)
    print(len(bin_edges))
    if trim:
        for i in range(len(bin_edges)-1):
            try:
                idxs.append(
                    np.where(arr_y == bin_stat[i])[0][0])
            except IndexError:
                pass
        idxs = list(set(idxs))
    else:
        for i in range(len(bin_edges)-1):
            idxs.append(np.where(arr_y == bin_stat[i])[0])
        idxs = list(set([item for subl in idxs for item in subl]))

    #lst = list(range(49, 67))
    #lst1=list(range(49))

    print('printing iso.lats1',iso.lats1)
    # Return a trimmed isochrone
    lats1 = iso.lats1[:, idxs]
    print('trimmed isochrone lat1',lats1)
    lons1 = iso.lons1[:, idxs]
    print('trimmed isochrone lat2',lons1)
    azi12 = iso.azi12[:, idxs]
    s12 = iso.s12[:, idxs]
    azi02 = iso.azi02[idxs]
    print('idxs',idxs)

    print('before s02',iso.s02)
    s02 = iso.s02[idxs]
    print('printing before s02',iso.s02[45])
    print('printing after s02',s02)
    iso = iso._replace(lats1=lats1)
    iso = iso._replace(lons1=lons1)
    iso = iso._replace(azi12=azi12)
    iso = iso._replace(s12=s12)
    iso = iso._replace(azi02=azi02)
    iso = iso._replace(s02=s02)
    #print(iso)
    print('prune iso.lats1',iso.lats1,iso.lons1)

    try:
        geeky_file = open('wind-router-master/wind-router-master/windrouter/dict5.txt', 'wt')
        geeky_file.write(str(iso.lons1[:, 102]))
        geeky_file = open('wind-router-master/wind-router-master/windrouter/dict3.txt', 'wt')
        geeky_file.write(str(iso))
        geeky_file.close()

    except:
        print("Unable to write to file")

    return iso

# For Routing with isochron
def recursive_routing(iso1,
                      boat,
                      winds,
                      delta_time,
                      params,
                      verbose=False):
    """
    Progress one isochrone with pruning.

            Parameters:
                iso1 (Isochrone) - starting isochrone
                start_point (tuple) - starting point of the route
                end_point (tuple) - end point of the route
                x1_coords (tuple) - tuple of arrays (lats, lons)
                x2_coords (tuple) - tuple of arrays (lats, lons)
                boat (dict) - boat profile
                winds (dict) - wind functions
                start_time (datetime) - start time
                delta_time (float) - time to move in seconds
                params (dict) - isochrone calculation parameters

            Returns:
                iso (Isochrone) - next isochrone
    """
    # branch out for multiple headings
    lats = np.repeat(iso1.lats1, params['ROUTER_HDGS_SEGMENTS'] + 1, axis=1)
    print(iso1.lats1,'printing iso1.lats1')
    print('length of iso.lats1',len(iso1.lats1),(iso1.lats1).shape)
    print('axis1 of lats',iso1.lats1[0,:])

    print(lats,'branch out for multiple headings')

    lons = np.repeat(iso1.lons1, params['ROUTER_HDGS_SEGMENTS'] + 1, axis=1)
    azi12 = np.repeat(iso1.azi12, params['ROUTER_HDGS_SEGMENTS'] + 1, axis=1)
    s12 = np.repeat(iso1.s12, params['ROUTER_HDGS_SEGMENTS'] + 1, axis=1)
    start_lats = np.repeat(iso1.start[0], lats.shape[1])
    start_lons = np.repeat(iso1.start[1], lons.shape[1])



    # determine new headings - centered around gcrs X0 -> X_prev_step
    hdgs = iso1.azi02
    print('first hdgs',hdgs)
    delta_hdgs = np.linspace(
        -params['ROUTER_HDGS_SEGMENTS'] * params['ROUTER_HDGS_INCREMENTS_DEG'],
        +params['ROUTER_HDGS_SEGMENTS'] * params['ROUTER_HDGS_INCREMENTS_DEG'],
        params['ROUTER_HDGS_SEGMENTS'] + 1)
    delta_hdgs = np.tile(delta_hdgs, iso1.lats1.shape[1])
    print('delta hdgs',delta_hdgs)
    hdgs = np.repeat(hdgs, params['ROUTER_HDGS_SEGMENTS'] + 1)
    print('second hdgs',hdgs)
    hdgs = hdgs - delta_hdgs
    print('GRC isochrone',delta_hdgs,' third hdgs ',hdgs)
    # move boat with defined headings N_coords x (ROUTER_HDGS_SEGMENTS+1) times
    #print('lats',lats[0,:],'lons',lons[0,:],'hdgs',hdgs)
    print('winds', winds)
    move = move_boat_direct(lats[0, :], lons[0, :], hdgs,
                            boat, winds,
                            iso1.time1, delta_time,
                            verbose=True)
    # print('geod.inverse ', move)
    print('printing gcr',lats[0,:])
    #print('hello lats from iso1',iso1.lats[:, 22])
    print('printinf lats00', lats[0, :].shape)
    # print('t12 time inside move',move['t12'])
    # create new isochrone before pruning
    # lats = np.vstack((move['lats2'], lats))
    print('moved lats',lats)
    # lons = np.vstack((move['lons2'], lons))
    # print('moved lons', lons)
    # azi12 = np.vstack((move['azi1'], azi12))
    # s12 = np.vstack((move['s12'], s12))
    # print('move so2',move['s12'])

    print('s12',s12)

    #start_lats = np.repeat(iso1.start[0], lats.shape[1])
    #start_lons = np.repeat(iso1.start[1], lons.shape[1])
    # determine gcrs from start to new isochrone
    # print('printing move lats2',move['lats2'])
    # print('printing move lons2', move['lons2'])
    lats2=36.0
    lons2=29.8
    #bs = boat_speed_function(boat, winds)
    dist = delta_time * 2
    gcr = geod.direct(lats, lons, hdgs, dist)
    gcrs = geod.inverse(gcr['lat2'],gcr['lon2'],iso1.finish[0],iso1.finish[1])
    print('new gcr',gcrs) # Compute geodesic between start lats lon and moved lat lon

    # remove those which ended on land
    # for i in range(int((x2 - x1) / STEP) + 1): #62.3, 17.6, 59.5, 24.6
    #     try:
    #         x = x1 + i * STEP
    #         y = (y1 - y2) / (x1 - x2) * (x - x1) + y1
    #     except:
    #         continue
    #     is_on_land = globe.is_land(float(x), float(y))
    #     print(is_on_land)
    #     # if not is_on_land:
    #     # print("in water")
    #
    #     if is_on_land:
    #         # print("crosses land")
    #
    #         return True

    is_on_land = globe.is_land(lats2, lons2)
    #print('isonland',is_on_land)
    # z = globe.is_land(lats, lons)
    # print('value of z',type(z))
    # if z=='True':
    #     is_on_land = globe.is_land(move['lats2'], move['lons2'])
    #     print(is_on_land)

    #print('printing land0',z)
    gcrs['s12'][is_on_land] = 0   #to check
    print('test',gcrs['s12'][is_on_land])
    azi02 = gcrs['azi1'] +1
    s02 = gcrs['s12']
    print('printing s02',s02)



    iso2 = Isochrone(
        start=iso1.start,
        finish=iso1.finish,
        gcr_azi=iso1.gcr_azi,
        count=iso1.count+1,
        elapsed=iso1.elapsed+dt.timedelta(seconds=delta_time),
        time1=iso1.time1+dt.timedelta(seconds=delta_time),
        lats1=lats,
        lons1=lons,
        azi12=azi12,
        s12=s12,
        azi02=azi02,
        s02=s02
    )

    # ---- pruning isochrone ----

    # new gcr azimuth to finish from the current isochrone
    mean_dist = np.mean(iso2.s02)
    print('printing mean distance',mean_dist)
    print('printing gcr_azi', iso1.gcr_azi)

    gcr_point = geod.direct(
        [iso1.start[0]],
        [iso1.start[1]],
        iso1.gcr_azi, mean_dist)
    print('printing gcr point',gcr_point)
    new_azi = geod.inverse(
        gcr_point['lat2'],
        gcr_point['lon2'],
        [iso1.finish[0]],
        [iso1.finish[1]]
    )
    print('new_azilll',new_azi)
    azi0s = np.repeat(
        new_azi['azi1'],
        params['ISOCHRONE_PRUNE_SEGMENTS'] + 1)

    # determine bins
    delta_hdgs = np.linspace(
        -params['ISOCHRONE_PRUNE_SECTOR_DEG_HALF'],
        +params['ISOCHRONE_PRUNE_SECTOR_DEG_HALF'],
        params['ISOCHRONE_PRUNE_SEGMENTS']+1)   #-90,+90,181

    print('printing delta hdgs',delta_hdgs)
    bins = azi0s - delta_hdgs
    print('printingazimuth0s', azi0s)
    print('printingazimuth0s', len(azi0s))

    bins = np.sort(bins)
    print('printing bin',bins)


    iso2 = prune_isochrone(iso2, 'azi02', 's02', bins, True)

    print('iso2 lat1 lon1',iso2.lats1,iso2.lons1)  #
    print('printings02',s02)
    print('printing azimuthazi02',azi02)
    geeky_file = open('wind-router-master/wind-router-master/windrouter/dict6.txt', 'wt')
    geeky_file.write(str(iso2.lons1[:]))
    print('hehehe',len(iso2.lons1),(iso2.lons1).shape)


    try:
        geeky_file = open('wind-router-master/wind-router-master/windrouter/dict2.txt', 'wt')
        geeky_file.write(str(iso2))
        geeky_file.close()

    except:
        print("Unable to write to file")

    return iso2


def routing(start,
            finish,
            boat,
            winds,
            start_time,
            delta_time,
            steps,
            params,
            verbose=False):
    """
    Do full isochrone routing.

            Parameters:
                iso1 (Isochrone) - starting isochrone
                start_point (tuple) - starting point of the route
                end_point (tuple) - end point of the route
                x1_coords (tuple) - tuple of arrays (lats, lons)
                x2_coords (tuple) - tuple of arrays (lats, lons)
                boat (dict) - boat profile
                winds (dict) - wind functions
                start_time (datetime) - start time
                delta_time (float) - time to move in seconds
                params (dict) - isochrone calculation parameters

            Returns:
                iso (Isochrone) - next isochrone
    """
    gcr = geod.inverse([start[0]], [start[1]], [finish[0]], [finish[1]])
    print('finish-',np.array([finish[0]]))
    print('first gcr and azimuth',gcr)
    #iso.lats1,iso.lons1=iso.lons1,iso.lats1
    iso3 = Isochrone(
        count=0,
        start=start,
        finish=finish,
        gcr_azi=gcr['azi1'],
        lats1=np.array([[start[0]]]),
        lons1=np.array([[start[1]]]),
       #lats2=np.array([[finish[0]]]),
       #lons2=np.array([[finish[1]]]),
        azi12=np.array([[None]]),
        s12=np.array([[0]]),
        azi02=gcr['azi1'],
        s02=np.array([]),
        time1=start_time,
        elapsed=dt.timedelta(seconds=0)
    )




    print('ISO Wrapper',iso3)



    for i in range(110):
        print('i in range ',i,' of ',110)
        iso3 = recursive_routing(
            iso3, boat, winds,
            delta_time, params,

            verbose=False)
        #print('printing params',params)

    try:
        geeky_file = open('wind-router-master/wind-router-master/windrouter/dict1.txt', 'wt')
        geeky_file.write(str(iso3))
        geeky_file.close()

    except:
        print("Unable to write to file")
    print('deleting soon final iso',iso3.lats1,iso3.lons1)
    return iso3


#
# def Squat(h, T, V_max, LWL, WWL, ukc, WVPI):
#     """Function for reducing the sailing velocity in deep water to the sailing velocity in shallow unconfined waters.
#     h:                              Array of the water depth in meters
#     V_max:                          Sailing velocity in deep water in meters per second
#     WWL:                            Width over Water Line of the vessel in meters
#     LWL:                            Length over Water Line of the vessel in meters
#     ukc:                            Minimal needed under keel clearance in  meters.
#     T:                              numpy array with the draft of the vessel. Numpy array has the shape of
#                                     the number of discretisations in the dynamic sailing velocity in meters
#     WVPI:                           total weight of the the vessel in tf
#     V:                              Array of sailing velocities reduced for squat, corresponding to the input arrat h.
#     """
#     Gamma_water = 1025
#     b = 9 * WWL
#     g = 9.81
#     ghv2 = g * h / (V_max ** 2)
#     squat_max = h - T - ukc
#     CB = WVPI / (LWL * WWL * T * Gamma_water)
#     AsAc = (WWL * T) / (b * h - WWL * T)
#     RH = (b * h - WWL * T) / (b + 2 * h + 2 * T + WWL)
#
#     VhV1 = (
#         1.002
#         + 0.005 * (np.sqrt(WWL * T) / RH)
#         - 0.1159 * (np.sqrt(WWL * T) / RH) ** 2
#         + 0.0191 * (np.sqrt(WWL * T) / RH) ** 3
#     )
#     V1Vinf = ((np.exp(ghv2) - np.exp(-ghv2)) / (np.exp(ghv2) + np.exp(-ghv2))) ** 0.5
#
#     V_grens = V_max * V1Vinf * VhV1
#     V_squat_max = np.zeros(len(h))
#     V = np.zeros(len(h))
#     for i in range(len(h)):
#         V_squat_max[i] = (
#             0
#             if squat_max[i] < 0
#             else (squat_max[i] * 30 / CB / (AsAc[i] ** (2 / 3))) ** (1 / 2.08)
#         )
#         V[i] = V_grens[i] if V_squat_max[i] > V_grens[i] else V_squat_max[i]
#
#     return V

