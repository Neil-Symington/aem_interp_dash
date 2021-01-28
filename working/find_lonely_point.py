import netCDF4
import os
import numpy as np
from garjmcmctdem_utils import spatial_functions

infile = "/home/nsymington/Documents/GA/dash_data_Surat/Injune_rjmcmc_pmaps.nc"
rj = netCDF4.Dataset(infile)

rj_coords = np.column_stack((rj['easting'][:],
                          rj["northing"][:]))

infile = "/home/nsymington/Documents/GA/dash_data_Surat/AUS_10024_InJune_EM_MGA55.nc"
em = netCDF4.Dataset(infile)

# find the big long line,which we want to ignore

long_line = 200001

line_index_mask = np.where(np.logical_and(np.where(em['line'][:] < 912999, True, False),
                           np.where(em['line'][:] != long_line, True, False)))[0]

line_mask = np.where(np.isin(em['line_index'][:], line_index_mask))[0]

em_coords = np.column_stack((em['easting'][line_mask],
                             em["northing"][line_mask],
                             em['fiducial'][line_mask])) # to keep track of points

fids = []

for i in range(200):
    dist, inds = spatial_functions.nearest_neighbours( em_coords[:,:2], rj_coords, max_distance = 1000000.)
    furthest_coord = em_coords[np.argmax(dist)]
    fids.append(furthest_coord[2])
    # NOw add the furthest coord to the rj_cords and repeat
    rj_coords = np.concatenate((rj_coords, np.array([furthest_coord[:2]])), axis = 0)

# Now we go through the inversion ready file and keep lines if the fid is in fids
infile = "/home/nsymington/Documents/GA/AEM/inversion_ready/Injune_inversion_ready.dat"

outfile = "/home/nsymington/Documents/GA/AEM/inversion_ready/Injune_inversion_ready_lonely_points.dat"

with open(infile, 'r') as f:
    with open(outfile, 'w') as outf:
        for line in f:
            list = line.split()
            fid = np.float(list[4])
            if fid in fids:
                outf.write(line)




