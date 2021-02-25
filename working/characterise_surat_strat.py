'''
This script demonstrates using interpreted points and probablistic AEM inversions to characterise the electrical
properties of stratigraphy from the Surat Basin

'''

import numpy as np
import pandas as pd
from garjmcmctdem_utils import spatial_functions, aem_utils, netcdf_utils
from scipy import interpolate
import netCDF4
import os
import matplotlib.pyplot as plt

infile = "/home/nsymington/Documents/GA/dash_data_Surat/200401_interp.csv"

df_interp = pd.read_csv(infile)

# bring in the rj inversion
nc_file = "/home/nsymington/Documents/GA/dash_data_Surat/Injune_pmaps_reduced_concatenated.nc"

rj = aem_utils.AEM_inversion(name = 'garjmcmctdem',  inversion_type = 'stochastic',
                              netcdf_dataset = netCDF4.Dataset(nc_file))

# do a nearest neighbour serach using our interpreted points
distances, indices = spatial_functions.nearest_neighbours(df_interp[['X','Y']].values, rj.coords, points_required=1)

unique_inds = np.unique(indices)

# iterate through these points, extract the surfaces and the conductivities

# results will be added to a data frame

cols = ['rj_index', 'easting', 'northing']

df_results = pd.DataFrame(columns = cols)
df_results['rj_index'] = unique_inds
df_results['easting'] = rj.data['easting'][unique_inds].data
df_results['northing'] = rj.data['northing'][unique_inds].data

# subset the interpretations into dataframes
df_interp_EvgPrecMool = df_interp[df_interp['BoundaryNm'] == "BaseEvergreenTopPrecipice"]
df_interp_PrecMool = df_interp[df_interp['BoundaryNm'] == "BasePrecipice_TopMoolyamber"]

interpolators = {}

# Create a linear interpolation for each
for item in ['BaseEvergreenTopPrecipice', 'BasePrecipice_TopMoolyamber']:
    df_interp_ss = df_interp[df_interp['BoundaryNm'] == item].sort_values('fiducial')
    # create an interpolation linear interpolation function in depth space
    f = interpolate.interp1d(df_interp_ss['Y'], # north south line
                             df_interp_ss['DEPTH'], # working in depth space
                             kind = 'linear', # splines return negative values
                             fill_value=np.nan,
                             bounds_error = False)
    df_results[item] = f(df_results['northing'])

df_results.dropna(how = 'any', inplace = True)

# Here we define the nominal top of Evergreen and base of Moolyamber for assigning characterisic values
# Note that this is in no way a mapped interface

df_results['baseMoolyamber'] = df_results["BasePrecipice_TopMoolyamber"] + 10.
df_results['topEvergreen'] = df_results["BaseEvergreenTopPrecipice"] - 10.

# Avoid negative depths
df_results['topEvergreen'] = df_results['topEvergreen'].where(df_results['topEvergreen'] > 0., 0.)

# create our conductivity histogram arrays
evergreen_cond_histogram = np.zeros(shape = rj.data.dimensions['conductivity_cells'].size,
                                    dtype = int)
precipice_cond_histogram = np.zeros(shape = rj.data.dimensions['conductivity_cells'].size,
                                    dtype = int)
moolyamber_cond_histogram = np.zeros(shape = rj.data.dimensions['conductivity_cells'].size,
                                     dtype = int)

# iterate through the dataframe and find the conductivity histogram for the ranges
for index, row in df_results.iterrows():
    rj_ind = int(row['rj_index'])
    # Evergreen
    mask = np.logical_and(row['topEvergreen'] < rj.data['layer_centre_depth'][:],
                          row['BaseEvergreenTopPrecipice'] > rj.data['layer_centre_depth'][:])
    evergreen_cond_histogram += rj.data['log10conductivity_histogram'][rj_ind, np.where(mask)[0], :].sum(axis = 0).astype(int)
    # Precipice
    mask = np.logical_and(row['BaseEvergreenTopPrecipice'] < rj.data['layer_centre_depth'][:],
                          row['BasePrecipice_TopMoolyamber'] > rj.data['layer_centre_depth'][:])
    precipice_cond_histogram += rj.data['log10conductivity_histogram'][rj_ind, np.where(mask)[0], :].sum(axis=0).astype(int)
    # Moolyamber
    mask = np.logical_and(row['BasePrecipice_TopMoolyamber'] < rj.data['layer_centre_depth'][:],
                          row['baseMoolyamber'] > rj.data['layer_centre_depth'][:])
    moolyamber_cond_histogram += rj.data['log10conductivity_histogram'][rj_ind, np.where(mask)[0], :].sum(axis=0).astype(int)



# Create a histogram plot
fig, ax_array = plt.subplots(3,1, sharex = True)
counts_, bins_, _ = ax_array[0].hist(rj.data['conductivity_cells'][:], bins=len(evergreen_cond_histogram),
                                     weights=evergreen_cond_histogram.astype(float)/np.sum(evergreen_cond_histogram),
                                     range=(rj.data['conductivity_cells'][0], rj.data['conductivity_cells'][-1]),
                                     color = 'powderblue')
ax_array[0].set_title('Evergreen Formation')
counts_, bins_, _ = ax_array[1].hist(rj.data['conductivity_cells'][:], bins=len(precipice_cond_histogram),
                                     weights=precipice_cond_histogram.astype(float)/np.sum(precipice_cond_histogram),
                                     range=(rj.data['conductivity_cells'][0], rj.data['conductivity_cells'][-1]),
                                     color = 'royalblue')
ax_array[1].set_title('Precipice Sandstone')
counts_, bins_, _ = ax_array[2].hist(rj.data['conductivity_cells'][:], bins=len(moolyamber_cond_histogram),
                                     weights=moolyamber_cond_histogram.astype(float)/np.sum(moolyamber_cond_histogram),
                                     range=(rj.data['conductivity_cells'][0], rj.data['conductivity_cells'][-1]),
                                     color = 'mediumspringgreen')
ax_array[2].set_title('Moolyamber Formation')
ax_array[2].set_xlabel('conductivity (S/m)')
ax_array[1].set_ylabel('probability')

#ax_array[0].set_xticks(10**ax_array[0].get_xticks())
#ax_array[1].set_xticks(10**ax_array[1].get_xticks())
#for i in range()
ax_array[2].set_xticklabels(['{:.3f}'.format(10**x) for x in ax_array[2].get_xticks()])
#print(ax_array[2].get_xticks())
ax_array[0].grid(True)
ax_array[1].grid(True)
ax_array[2].grid(True)
plt.savefig("Surat_line_200401_histogram.png", dpi = 200)
plt.show()

