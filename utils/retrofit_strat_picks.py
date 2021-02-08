# This script takes existing stratigraphic interpretations and uses the pmaps to make the points more consistent with
# the AEM model ensemble. It also assigns uncertainty using the full width half maximum technique

import netCDF4
import pandas as pd
import numpy as np
import os
from garjmcmctdem_utils import spatial_functions, aem_utils

root = r"E:\GA\dash_data_Surat"

# From the app TODO add to the garjmcmctdem_scripts
def full_width_half_max(interpreted_depth, depth_array, count_array, snap_window = 20.):
    """
    Function for calculating the interval that is >0.5 times the probability
    Parameters
    ----------
    interpreted_depth: float of the interpreted depth
    depth_array: array of depths
    count_array: array of counts. nb this is a proxy for probabilities

    Returns
    -------
    tuple of indices denoting the top and bottom of the full width half max
    """

    idx_upper = -1
    idx_lower = -1

    # Find the maximum depth index
    max_idx = np.argmin(np.abs(depth_array - interpreted_depth))

    # get the maximum probability
    fmax = count_array[max_idx]

    # Snap to maximum likelihood depth
    # find the search window
    window = int(np.ceil((snap_window/2.)/2))

    window_slice = [max_idx - window, max_idx + window]
    # to ensure no index errors
    if window_slice[0] < 0:
        window_slice[0] = 0
    if window_slice[1] >= len(count_array):
        window_slice[1] = len(count_array)

    # positive direction

    for idx in np.arange(max_idx, depth_array.shape[0]):
        if count_array[idx] <= fmax/2.:
            idx_upper = idx
            break
    # negative direction
    for idx in np.arange(max_idx, -1, -1):
        if count_array[idx] <= fmax/2.:
            idx_lower = idx
            break
    # Now calculate the width
    return (idx_lower, max_idx, idx_upper)

stratPicksFile = os.path.join(root,
                              "Surat_basin_AEM_interpretations.csv")

df_strat = pd.read_csv(stratPicksFile)

strat_coords = df_strat[['X', "Y"]]

# open pmaps

pmap_file =os.path.join(root,"Injune_pmaps_reduced_concatenated.nc")

rj = aem_utils.AEM_inversion(name = 'garj',
                              inversion_type = 'stochastic',
                              netcdf_dataset = netCDF4.Dataset(pmap_file))

# Find the nearest neighbours for our interpretation points

dist, ind = spatial_functions.nearest_neighbours(strat_coords, rj.coords)

df_strat['rj_ind'] = ind

# now iterate through each point, snap to the maximum likelihood if it exists

uncerts = np.nan*np.ones(shape = ind.shape, dtype = np.float)
new_depth = np.nan*np.ones(shape = ind.shape, dtype = np.float)

depths = df_strat['DEPTH'].values

# garjmcmctdem depths
depth_array = rj.data['layer_centre_depth'][:] - 1.

for i, rj_ind in enumerate(ind):
    # get the count array
    count_array = rj.data['interface_depth_histogram'][rj_ind]
    interpreted_depth = depths[i]

    # Make our maximum uncertainty equal to the interpretation depth. This is arbitrary and needs to be tested
    max_uncert = interpreted_depth

    # We will make our snap wind to 50% of the depth
    snap_window = interpreted_depth/2.

    idx_lower, max_idx, idx_upper = full_width_half_max(interpreted_depth, depth_array,
                                                        count_array, snap_window = snap_window)

    # add these to the the arrays
    if idx_upper != -1 and idx_lower != -1:
        uncertainty = depth_array[idx_upper] - depth_array[idx_lower]
        if uncertainty < max_uncert:
            new_depth[i] = depth_array[max_idx]
            uncerts[i] = uncertainty

# Add these to the dataframe

df_strat['DEPTH_UNCERTAINTY'] = uncerts

for i, d in enumerate(new_depth):
    if not np.isnan(d):
        df_strat.at[i, "DEPTH"] = d

stratPicksOutfile = os.path.join(root,"Surat_basin_AEM_interpretations_uncerts.csv")
df_strat.to_csv(stratPicksOutfile)