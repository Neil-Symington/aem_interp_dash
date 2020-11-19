#!/usr/bin/env python

#===============================================================================
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
#===============================
from collections import Sequence
import h5py
import numpy as np
import xarray as xr
import pickle

def check_list_arg(object):
    """Function for checking if

    Parameters
    ----------
    object : object
        Python object.
    Returns
    -------
    list object

    """
    if isinstance(object, (Sequence, np.ndarray)):
        return object
    else:
        return [object]

def dict_to_hdf5(fname, dictionary):
    """
    Save a dictionary to hdf5
    """
    f = h5py.File(fname, "w")
    for key in dictionary.keys():
        dset = f.create_dataset(key, data=dictionary[key])
    f.close()

def extract_hdf5_data(file, grid_vars):
    """Short summary.

    Parameters
    ----------
    file : type
        Description of parameter `file`.
    grid_vars : type
        Description of parameter `grid_vars`.

    Returns
    -------
    type
        Description of returned object.

    """

    datasets = {}

    for item in file.values():
        if item.name[1:] in grid_vars:
            datasets[item.name[1:]] = item[()]
        # We also need to know easting, northing, doi, elevations and grid elevations
        if item.name[1:] == 'easting':
            datasets['easting'] = item[()]
        if item.name[1:] == 'northing':
            datasets['northing'] = item[()]
        if item.name[1:] == 'grid_elevations':
            datasets['grid_elevations'] = item[()]
        if item.name[1:] == 'depth_of_investigation':
            datasets['depth_of_investigation'] = item[()]
        if item.name[1:] == 'elevation':
            datasets['elevation'] = item[()]
        if item.name[1:] == 'grid_distances':
            datasets['grid_distances'] = item[()]

    return datasets

def dict2xr(d, dims=None):
    # define coordinates
    if dims is None:
        dims = ['grid_elevations', 'grid_distances']
    var_list = [e for e in d.keys() if e not in dims]
    data_vars = {}
    coords = {}

    # Now we find the dimensions
    for item in dims:
        coords[item] = d[item]

    for item in var_list:
        if len(d[item].shape) == 2:
            if len(dims) == 2:
                data_vars[item] = (dims, d[item])
            elif len(dims) == 1:
                new_dim = item + '_dim'
                new_dims = dims + [new_dim]
                data_vars[item] = (new_dims, d[item])
                coords[new_dim] = np.arange(0,d[item].shape[1])
        elif len(d[item].shape) == 1:
            data_vars[item] = (dims[-1], d[item])

    ds = xr.Dataset(data_vars, coords=coords)
    return ds


def pickle2xarray(infile):

    with open(infile, 'rb') as file:
        xarr = pickle.load(file)
    return xarr

def xarray2pickle(xarr, outfile):
    file = open(outfile, 'wb')
    # dump information to the file
    pickle.dump(xarr, file)
