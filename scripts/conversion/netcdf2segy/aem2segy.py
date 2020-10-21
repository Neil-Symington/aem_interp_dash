#!/usr/bin/env python

# ===============================================================================
#    Copyright 2017 Geoscience Australia
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ===============================================================================

'''
Created on 8/5/2019
@author: Neil Symington
Functions for converting the aseg gdf data to seg-y
'''

import numpy as np
from scipy import interpolate


# Function for nulling all values below the doi
def remove_below_doi(interpolated_data, z_new, doi, elevation):
    """
    :param interpolated_data: numpy array with interpolated data
    :param z_new: new elevation intervals for segy trace
    :param doi: float with fiducial depth of investigation
    :param elevation: float fiducial with elevation
    :return:
    interpolated_data with below doi values changed to -1.
    """

    doi_elevation = -1 * (elevation - doi)

    # Find the indices that are below the depth of investigation
    interpolated_data[np.where(z_new > doi_elevation)] = -1

    return interpolated_data


# Interpolate so that we have a continuously spaced data

def interpolate_layer_data(depth_top, z_new, dat, elev, max_depth, datum):
    # First find layer bottom (by adding a small delta d)

    depth_bottom = depth_top[1:] - 0.01

    # Now add the layer tops and bottoms into a single array and produce a
    # corresponding conductivity array

    # The aim is to book end each layer
    z = []
    new_dat = []
    for i in range(len(depth_bottom)):
        z.append(depth_top[i])
        z.append(depth_bottom[i])
        new_dat.append(dat[i])
        new_dat.append(dat[i])

    # Convert the depth to elevation (where negative values are above msl)
    z = [x - elev for x in z]

    # Finally bookend the air and give it a conductivity of 0

    z.insert(0, z[0] - 0.01)
    z.insert(0, datum * -1)

    new_dat.insert(0, -1)
    new_dat.insert(0, -1)

    # Now bookend the bottom half-space to the max depth
    z.append(z[-1] + 0.01)
    z.append(-1 * max_depth * -1)

    new_dat.append(dat[-1])
    new_dat.append(dat[-1])

    f = interpolate.interp1d(z, new_dat)

    interpolated_dat = f(z_new)

    return interpolated_dat

def get_line_mask(line, netCDF_dataset):
    """A function for return a mask for an AEM line/ lines

    Parameters
    ----------
    line : int
        line number
    netCDF_dataset:
        netcdf dataset with variables 'line' and 'line_index'

    Returns
    -------
    self, boolean array
        Boolean mask for lines

    """

    line_inds = np.where(np.isin(netCDF_dataset['line'][:], line))[0]

    return np.isin(netCDF_dataset['line_index'],line_inds)

def get_sorted_line_data(netCDF_dataset, line, vars, sort_on = None):
    # get line mask
    line_mask = get_line_mask(line, netCDF_dataset)

    # iterate through the variables and get them into a data dictionary
    data_dict = {}

    for item in vars:
        data_dict[item] = netCDF_dataset.variables[item][line_mask]

    # Now sort on a variable if desired
    if sort_on is not None:
        sort_mask = np.argsort(data_dict[sort_on])
        for item in vars:
            data_dict[item] = data_dict[item][sort_mask]

    return data_dict

