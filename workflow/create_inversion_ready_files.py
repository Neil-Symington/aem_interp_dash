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
sys.path.append(r"C:/Users/symin/github/garjmcmctdem_utils/scripts")
import shapely
import netCDF4
import aem_utils
import netcdf_utils
import spatial_functions

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


infile = r"C:\Users\symin\OneDrive\Documents\GA\AEM\EM\AUS_10023_SSC_EM_MGA53.nc"

injune = aem_utils.AEM_data(name = 'Southern Stuart Corridor AEM data',
                              system_name = 'SkyTEM312',
                              netcdf_dataset = netCDF4.Dataset(infile))

# use inbuilt functions to calculate the noise
injune.calculate_noise("low_moment_Z-component_EM_data", noise_variable = "low_moment_Z_component_noise",
                       multiplicative_noise = 0.03)

injune.calculate_noise("high_moment_Z-component_EM_data", noise_variable = "high_moment_Z_component_noise",
                       multiplicative_noise = 0.03)

# Noise is root of sum of the squares
lm_noise = injune.low_moment_Z_component_noise
hm_noise = injune.high_moment_Z_component_noise

# Now divide this by the absolute value of the data to get the relative uncertainty

lm_runc = lm_noise/np.abs(injune.data['low_moment_Z-component_EM_data'][:].data)
hm_runc = hm_noise/np.abs(injune.data['high_moment_Z-component_EM_data'][:].data)

# NOw we run a spatial mask to find the appropriate sites

# First we parse the interp file and extract point coordinates

interp_coords = np.empty(shape=[0, 3])

for file in glob.glob(os.path.join(r"C:\Users\symin\github\garjmcmctdem_utils\data", '*')):
    # Only interested in files with no extension
    if os.path.splitext(os.path.basename(file))[-1] == '':
        # just get the line number and coordinates for now
        interp_coords= np.append(interp_coords, np.loadtxt(file, skiprows = 1, usecols = [0, 2,3]), axis = 0)

# Now lets do a spatial search on our AEM coordinates
aem_coords = injune.coords

dist, inds = spatial_functions.nearest_neighbours(interp_coords[:,1:], aem_coords)

final_inds = np.unique(inds)#[::5]

# Create a boolean mask array

mask_array = np.zeros(injune.data.dimensions['point'].size, dtype=np.bool)

#mask_array[final_inds] = True

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
outfile = r"C:\Users\symin\OneDrive\Documents\GA\AEM\inversion_ready\SSC_inversion_ready.dat"

# Write the inversion file to disc
write_inversion_ready_file(injune.data, outfile, nc_variables,
                               nc_formats, other_variables = other_data,
                               mask = None)#mask_array)

# Here we write the additive noise to a file. This allows us to copy and past it into a control file,
# which is required by some inversions

outfile = r"C:\Users\symin\OneDrive\Documents\GA\AEM\inversion_ready\SSC_hm_additive_noise.txt"

with open(outfile, 'w') as f:
    f.write('\t'.join(map(str,injune.high_moment_Z_component_EM_data_additive_noise)))

outfile = r"C:\Users\symin\OneDrive\Documents\GA\AEM\inversion_ready\SSC_lm_additive_noise.txt"

with open(outfile, 'w') as f:
    f.write('\t'.join(map(str,injune.low_moment_Z_component_EM_data_additive_noise)))

injune.data.close()