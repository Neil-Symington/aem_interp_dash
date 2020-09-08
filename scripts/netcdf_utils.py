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
#===============================================================================

'''
Created on 7/7/2019
@author: Neil Symington

Utility functions for dealing with netcdf data

'''

import netCDF4
import numpy as np
import spatial_functions
import pandas as pd

def object2array(variable, dtype):
    """Helper function for converting single variables to a list

    Parameters
    ----------
    variable : object
        Python object
    dtype : python datatype

    Returns
    -------
    list
        If variable is a single variables return the variable within a list

    """

    single_var = (type(variable) == dtype)
    if single_var:
        return [variable]
    else:
        return variable

def get_lines(dataset, line_numbers, variables):
    """
    A function for extracting variables from a particular AEM line
    @param dataset: netcdf dataset
    @param: list of integer AEM line_numbers:
    @param: list of integer variables)
    """
    # Allow single variable to be given as a string

    variables = object2array(variables, str)
    # Allow single line

    line_numbers = object2array(line_numbers, int)

    # Chekc netcdf dataset
    if not dataset.__class__ == netCDF4._netCDF4.Dataset:
        raise ValueError("Input datafile is not netCDF4 format")
        return None

    # Iterate through lines and get the point indices
    for line in line_numbers:
        point_mask = mask = dataset['line_index'][:] == np.where(dataset['line'][:] == line)[0]
        # Iterate through the variables and add the masked arrays to a dictionary
        line_dict = {}

        for var in variables:
            line_dict[var] = dataset[var][point_mask]

        yield line, line_dict

def extract_rj_sounding(rj, lci, point_index = 0):
    """
    TODO: clean up this function or consider removing!!!
    """

    rj_dat = rj.data
    lci_dat = lci.data

    freq = rj_dat['log10conductivity_histogram'][point_index].data.astype(np.float)

    easting = np.float(rj_dat['easting'][point_index].data)
    northing = np.float(rj_dat['northing'][point_index].data)

    cond_pdf = freq / freq.sum(axis =1)[0]

    cond_pdf[cond_pdf == 0] = np.nan

    cp_freq = rj_dat["interface_depth_histogram"][point_index].data.astype(np.float)

    cp_pdf = cp_freq / freq.sum(axis =1)[0]

    laybins = rj_dat['nlayers_histogram'][point_index].data

    lay_prob = laybins / freq.sum(axis =1)[0]

    condmin, condmax = rj_dat.min_log10_conductivity, rj_dat.max_log10_conductivity

    ncond_cells = rj_dat.dimensions['conductivity_cells'].size

    cond_cells = np.linspace(condmin, condmax, ncond_cells)

    pmin, pmax = rj_dat.min_depth, rj_dat.max_depth

    depth_cells = rj_dat['layer_centre_depth'][:]

    extent = [cond_cells.min(), cond_cells.max(), depth_cells.max(), depth_cells.min()]

    mean = np.power(10,rj_dat['conductivity_mean'][point_index].data)
    p10 = np.power(10,rj_dat['conductivity_p10'][point_index].data)
    p50 = np.power(10,rj_dat['conductivity_p50'][point_index].data)
    p90 = np.power(10,rj_dat['conductivity_p90'][point_index].data)

    lci_coords = np.column_stack

    distances, indices = spatial_functions.nearest_neighbours([easting, northing],
                                                              lci.coords,
                                                               max_distance = 100.)
    point_ind_lci = indices[0]

    lci_cond = lci_dat['conductivity'][point_ind_lci].data
    lci_depth_top = lci_dat['layer_top_depth'][point_ind_lci].data

    lci_doi = lci_dat['depth_of_investigation'][point_ind_lci].data

    misfit = np.sqrt(rj_dat['misfit'][point_index].data)

    burnin = rj_dat.nburnin
    nsamples = rj_dat.nsamples
    sample_no = np.arange(1,rj_dat.dimensions['convergence_sample'].size + 1)
    nchains = rj_dat.nchains
    elevation = rj_dat['elevation'][point_index]
    line = int(rj_dat['line'][point_index])
    fiducial = float(rj_dat['fiducial'][point_index])
    elevation = rj_dat['elevation'][point_index]

    dist = spatial_functions.xy_2_var(lci.section_data[line],
                                      np.array([[easting, northing]]),
                                      'grid_distances')

    return {'conductivity_pdf': cond_pdf, "change_point_pdf": cp_pdf, "conductivity_extent": extent,
           'cond_p10': p10, 'cond_p50': p50, 'cond_p90': p90, 'cond_mean': mean, 'depth_cells': depth_cells,
           'nlayer_bins': laybins, 'nlayer_prob': lay_prob, 'nsamples': nsamples, 'ndata': rj_dat.dimensions['data'].size,
           "nchains": nchains, 'burnin': burnin, 'misfit': misfit, 'sample_no': sample_no, 'cond_cells': cond_cells, 'lci_cond': lci_cond,
           'lci_depth_top': lci_depth_top, 'lci_doi': lci_doi, 'line': line, 'northing': northing, 'easting': easting, 'fiducial':fiducial,
           'elevation': elevation, 'lci_dist': dist, 'lci_line': lci.section_data[line]}

def testNetCDFDataset(netCDF_dataset):
    """Test if datafile is netcdf.
    TODO add a check of necessary parameters

    Parameters
    ----------
    netCDF_dataset : object
        netcdf AEM dataset.

    Returns
    -------
    boolean

    """


    return netCDF_dataset.__class__ == netCDF4._netCDF4.Dataset

def get_lookup_mask(lines, netCDF_dataset):
    """A function for return a mask for an AEM line/ lines

    Parameters
    ----------
    lines : array like
        array of line numbers
    netCDF_dataset:
        netcdf dataset with variables 'line' and 'line_index'

    Returns
    -------
    self, boolean array
        Boolean mask for lines

    """
    lines = object2array(lines, int)

    line_inds = np.where(np.isin(netCDF_dataset['line'][:], lines))[0]

    return np.isin(netCDF_dataset['line_index'],line_inds)

def write_inversion_ready_file(dataset, outpath, nc_variables,
                               nc_formats, other_variables = None,
                               mask = None):
    """A function for writing an inversion ready.dat file. This file can be
     inverted using GA-AEM inversion algorithms.

    Parameters
    ----------
    dataset : object
        Netcdf dataset
    outpath : string
        Path of inversion ready file.
    nc_variables : list
        List of variables from dataset.
        eg ["ga_project", "utc_date", "flight", "line", "fiducial",
            "easting", "northing", "tx_height_measured", "elevation",
            "gps_height", "roll", "pitch", "yaw", "TxRx_dx", "TxRx_dy",
            "TxRx_dz", "low_moment_Z-component_EM_data",
            "high_moment_Z-component_EM_data"]
    nc_formats : list
        List of formats for variables.
        eg  ['{:5d}','{:9.0F}','{:12.2F}','{:8.0F}','{:12.2F}','{:10.2F}',
             '{:11.2F}','{:8.1F}','{:9.2F}', '{:9.2F}','{:7.2F}','{:7.2F}',
             '{:7.2F}','{:7.2F}','{:7.2F}','{:7.2F}', '{:15.6E}', '{:15.6E}']
    other_variables : dictionary
        dictionary of additional variables with the name of the variable as the
        key, 'array' key as the
        e.g.{'rel_uncertainty_low_moment_Z-component':
             {'data': numpy array, 'format': {:15.6E}} }
    mask: boolean array

    """
    # Now create a mask if none exists
    if mask is None:
        mask = np.ones(shape = (dataset.dimensions['point'].size), dtype = np.bool)
    # Create an empty dataframe
    df = pd.DataFrame(index = range(mask.sum()))

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

        if len(arr.shape) < 2:
            df[item] = [data[item]['format'].format(x) for x in arr]
        # For 3d variables like the EM data
        else:
            for i in range(arr.shape[1]):
                df[item + '_' + str(i+1)] = [data[item]['format'].format(x) for x in arr[:,i]]
    # Note use a pipe so we can easily delete later

    df.apply(lambda row: ''.join(map(str, row)), axis=1).to_csv(outpath, sep = ',', index = False, header = False)

    # Now write the .hdr file
    header_file = '.'.join(outfile.split('.')[:-1]) + '.hdr'
    counter = 1
    with open(header_file, 'w') as f:
        for item in data.keys():
            shape = data[item]['array'].shape
            if len(shape) == 1:
                f.write(''.join([item, ' ', str(counter), '\n']))
                counter += 1
            else:
                f.write(''.join([item,' ',str(counter),'-',str(counter + shape[1] - 1),'\n']))
                counter += shape[1]
