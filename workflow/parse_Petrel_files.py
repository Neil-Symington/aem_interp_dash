'''
Created on 16/6/2020
@author: Neil Symington

This script is for parsing geological interpretations from atext file exported from Petrel and writing it into a
series of csv files
'''

import pandas as pd
import numpy as np
import os, sys, glob
import netCDF4
sys.path.append(r"C:/Users/symin/github/garjmcmctdem_utils/scripts")
import aem_utils, spatial_functions

# First we bring in the lci data. This is so we can pair the points with a fiducial

root = r"C:\Users\symin\OneDrive\Documents\GA\AEM\LCI"

infile = os.path.join(root, "Injune_lci_MGA55.nc")

# Create an instance
lci = aem_utils.AEM_inversion(name = 'Laterally Contrained Inversion (LCI)',
                              inversion_type = 'deterministic',
                              netcdf_dataset = netCDF4.Dataset(infile))


df = pd.DataFrame(columns = ['line', 'easting', 'northing', 'depth_mBGL', 'elevation_mAHD', 'interface', 'fiducial'])



for file in glob.glob(os.path.join(r"C:\Users\symin\github\garjmcmctdem_utils\data", '*')):
    # Only interested in files with no extension
    if os.path.splitext(os.path.basename(file))[-1] == '':
        # just get the line number and coordinates for now
        dat = np.loadtxt(file, skiprows = 1, usecols = [0,2,3,4])
        # nearest neightbor to get fiducials
        dist, inds = spatial_functions.nearest_neighbours(dat[:, 1:3], lci.coords)

        fids = lci.data['fiducial'][inds]

        ground_elevation = lci.data['elevation'][inds]

        interp_elevation = dat[:,3]
        interp_depth = ground_elevation - interp_elevation

        # Create temporary dataframe
        df_temp = pd.DataFrame(data = {'line': dat[:,0], 'easting':dat[:,1],
                                       'northing': dat[:,2], 'depth_mBGL': interp_depth,
                                       'elevation_mAHD': interp_elevation,
                                       'interface': os.path.basename(file),
                                       'fiducial': fids})
        # Append to main dataframe
        df = df.append(df_temp)

# Save interpretation as a csv

df.to_csv(r"C:\Users\symin\github\garjmcmctdem_utils\data\Surat_basin_AEM_interpretations.csv")



