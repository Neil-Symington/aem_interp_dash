'''
Created on 14/6/2020
@author: Neil Symington
This script is for creating inversion ready data files that can be used as inputs for inversion. This involves estimating
the noise for each AEM gate at every site. We export the inversion ready files at every site with a corresponding
interpretation.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os, glob
import shapely
import netCDF4
from garjmcmctdem_utils import aem_utils, netcdf_utils, spatial_functions

# Function for writing inversion file. WE recycle a .dfn

def write_inversion_ready_file(dataset, outpath, nc_variables,
                               nc_formats, other_variables=None,
                               mask=None):
    # Now create a mask if none exists
    if mask is None:
        mask = np.ones(shape=(dataset.dimensions['point'].size), dtype=np.bool)
    # Create an empty dataframe
    df = pd.DataFrame(index=range(mask.sum()))

    # Create a dictionary with arrays, formats and variable name
    data = {}
    for i, var in enumerate(nc_variables):
        if var == 'line':
            line_inds = dataset['line_index'][mask]
            arr = dataset[var][line_inds].data
        elif var == 'flight':
            flight_inds = dataset['flight_index'][mask]
            arr = dataset[var][flight_inds].data
        # Scalar variables
        elif len(dataset[var].shape) == 0:
            arr = np.repeat(dataset[var][:].data, mask.sum())
        else:
            arr = dataset[var][mask].data
        # Add to dictionary
        data[var] = {'array': arr,
                     'format': nc_formats[i]}

    # Now we add the additional columns
    if other_variables is not None:
        for item in other_variables.keys():
            # apply mask
            data[item] = {'array': other_variables[item]['array'][mask],
                          'format': other_variables[item]['format']}
    # build pandas dataframe
    for item in data:
        print(item)
        arr = data[item]['array']
        print(arr.shape)
        if len(arr.shape) < 2:
            df[item] = [data[item]['format'].format(x) for x in arr]
        # For 3d variables like the EM data
        else:
            for i in range(arr.shape[1]):
                df[item + '_' + str(i + 1)] = [data[item]['format'].format(x) for x in arr[:, i]]
    # Note use a pipe so we can easily delete later

    df.apply(lambda row: ''.join(map(str, row)), axis=1).to_csv(outpath, sep=',',
                                                                index=False, header=False)

    # Now write the .hdr file
    header_file = '.'.join(outpath.split('.')[:-1]) + '.hdr'
    counter = 1
    with open(header_file, 'w') as f:
        for item in data.keys():
            shape = data[item]['array'].shape
            if len(shape) == 1:
                f.write(''.join([item, ' ', str(counter), '\n']))
                counter += 1
            else:
                f.write(''.join([item, ' ', str(counter), '-', str(counter + shape[1] - 1), '\n']))
                counter += shape[1]

infile ="/home/nsymington/Documents/GA/AEM/EM/AUS_10021_OrdK_EM.nc"

ek = aem_utils.AEM_data(name = 'East Kimberley AEM data',
                              system_name = 'SkyTEM312',
                              netcdf_dataset = netCDF4.Dataset(infile))


# use inbuilt functions to calculate the noise
ek.calculate_noise("low_moment_Z-component_EM_data", noise_variable = "low_moment_Z_component_noise",
                       multiplicative_noise = 0.03)

ek.calculate_noise("high_moment_Z-component_EM_data", noise_variable = "high_moment_Z_component_noise",
                       multiplicative_noise = 0.03)

# Noise is root of sum of the squares
lm_noise = ek.low_moment_Z_component_noise
hm_noise = ek.high_moment_Z_component_noise

# Now divide this by the absolute value of the data to get the relative uncertainty

lm_runc = lm_noise/np.abs(ek.data['low_moment_Z-component_EM_data'][:].data)
hm_runc = hm_noise/np.abs(ek.data['high_moment_Z-component_EM_data'][:].data)

# NOw we run a line mask

#lines = [503401]
#
line_mask = netcdf_utils.get_lookup_mask(lines, mus.data)

# Create a boolean mask array

mask_array = np.zeros(ek.data.dimensions['point'].size, dtype=np.bool)

# now we want to create a point mask
infile = "/home/nsymington/Documents/GA/EK_salinity_mapping/EK_inversions_AEGC/Keep_tight_pilot_points.csv"
df_pp = pd.read_csv(infile)

pp_coords = df_pp[['X', 'Y']]

dist, ind = spatial_functions.nearest_neighbours(pp_coords, ek.coords)

#mask_array[line_mask] = True
#mask_array[:] = True
mask_array[ind] = True

# Now write these data to a .dat file with the variables that are important for inversion.

nc_variables = ["ga_project", "utc_date", "flight", "line", "fiducial", "easting",
                "northing", "tx_height_measured", "elevation", "gps_height", "roll",
                "pitch", "yaw", "TxRx_dx", "TxRx_dy", "TxRx_dz",
                "low_moment_Z-component_EM_data","high_moment_Z-component_EM_data"]

nc_formats = ['{:5d}','{:9.0F}','{:12.2F}','{:8.0F}','{:12.2F}','{:10.2F}','{:11.2F}',
              '{:8.1F}','{:9.2F}','{:9.2F}','{:7.2F}','{:7.2F}','{:7.2F}','{:7.2F}',
              '{:7.2F}','{:7.2F}', '{:15.6E}', '{:15.6E}']
# Here we are adding our calculated relative uncertainty
other_data = {'rel_uncertainty_low_moment_Z-component': {'array': lm_runc, 'format': '{:15.6E}'},
              'rel_uncertainty_high_moment_Z-component': {'array': hm_runc, 'format': '{:15.6E}'}}

# Define the outfile
#outfile = r"C:\Users\u77932\Documents\EFTF2\SW\data\existing\AEM\110284_Data_Package\AEM\inversion_ready\Musgraves_inversion_ready.dat"
#outfile = r"C:\Users\u77932\Documents\EFTF2\SW\data\existing\AEM\110284_Data_Package\AEM\inversion_ready\Musgraves_line_503401_inversion_ready.dat"
outfile = "/home/nsymington/Documents/GA/EK_salinity_mapping/EK_inversions_AEGC/EK_pilot_points_inversion_ready.dat"

# Write the inversion file to disc
write_inversion_ready_file(ek.data, outfile, nc_variables,
                               nc_formats, other_variables = other_data,
                               mask = mask_array)

# Here we write the additive noise to a file. This allows us to copy and past it into a control file,
# which is required by some inversions

outfile = r"C:\Users\u77932\Documents\EFTF2\SW\data\existing\AEM\110284_Data_Package\AEM\inversion_ready\Musgraves_hm_additive_noise.txt"

#with open(outfile, 'w') as f:
#    f.write('\t'.join(map(str,mus.high_moment_Z_component_EM_data_additive_noise)))

outfile = r"C:\Users\u77932\Documents\EFTF2\SW\data\existing\AEM\110284_Data_Package\AEM\inversion_ready\Musgraves_lm_additive_noise.txt"

#with open(outfile, 'w') as f:
#    f.write('\t'.join(map(str,mus.low_moment_Z_component_EM_data_additive_noise)))

ek.data.close()