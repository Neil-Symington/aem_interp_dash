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

def get_lines(dataset, line_numbers, variables):
    """
    A function for extracting variables from a particular AEM line
    @param dataset: netcdf dataset
    @param: list of integer AEM line_numbers:
    @param: list of integer variables)
    """
    # Allow single variable to be given as a string
    single_var = (type(variables) == str)
    if single_var:
        variables = [variables]
    # Allow single line
    single_line = (type(line_numbers) == int)
    if single_line:
        line_numbers = [line_numbers]

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


def extract_rj_sounding(rj_dat, lci_dat, point_index = 0):
    """
    A function for extracting rj data into numpy arrays within a python dictionary
    """

    freq = rj_dat['log10conductivity_histogram'][point_index].data.astype(np.float)

    easting = np.float(rj_dat['x'][point_index].data)
    northing = np.float(rj_dat['y'][point_index].data)

    cond_pdf = freq / freq.sum(axis =1)[0]

    cond_pdf[cond_pdf == 0] = np.nan

    cp_freq = rj_dat["interface_depth_histogram"][point_index].data.astype(np.float)

    cp_pdf = cp_freq / freq.sum(axis =1)[0]

    laybins = rj_dat['nlayers_histogram'][point_index].data

    lay_prob = laybins / freq.sum(axis =1)[0]

    condmin, condmax = rj_dat.vmin, rj_dat.vmax

    ncond_cells = rj_dat.dimensions['value'].size

    cond_cells = np.linspace(condmin, condmax, ncond_cells)

    pmin, pmax = rj_dat.pmin, rj_dat.pmax


    depth_cells = rj_dat['depth'][point_index].data

    extent = [cond_cells.min(), cond_cells.max(), depth_cells.max(), depth_cells.min()]

    mean = np.power(10,rj_dat['mean_model'][point_index].data)
    p10 = np.power(10,rj_dat['p10_model'][point_index].data)
    p50 = np.power(10,rj_dat['p50_model'][point_index].data)
    p90 = np.power(10,rj_dat['p90_model'][point_index].data)

    lci_coords = np.column_stack((lci_dat['easting'][:],
                          lci_dat['northing'][:]))

    distances, indices = spatial_functions.nearest_neighbours([easting, northing], lci_coords, max_distance = 50.)

    point_ind_lci = indices[0]

    lci_cond = lci_dat['conductivity'][point_ind_lci].data
    lci_depth_top = lci_dat['layer_top_depth'][point_ind_lci].data

    lci_doi = lci_dat['depth_of_investigation'][point_ind_lci].data

    misfit = np.sqrt(rj_dat['misfit'][point_index].data)

    burnin = rj_dat.nburnin
    nsamples = rj_dat.nsamples
    sample_no = rj_dat['convergence_sample'][point_index].data
    nchains = rj_dat.nchains
    elevation = rj_dat['elevation'][point_index]
    line = int(rj_dat['line'][point_index])
    fiducial = float(rj_dat['fiducial'][point_index])
    elevation = rj_dat['elevation'][point_index]

    return {'conductivity_pdf': cond_pdf, "change_point_pdf": cp_pdf, "conductivity_extent": extent,
           'cond_p10': p10, 'cond_p50': p50, 'cond_p90': p90, 'cond_mean': mean, 'depth_cells': depth_cells,
           'nlayer_bins': laybins, 'nlayer_prob': lay_prob, 'nsamples': nsamples, 'ndata': rj_dat.dimensions['data'].size,
           "nchains": nchains, 'burnin': burnin, 'misfit': misfit, 'sample_no': sample_no, 'cond_cells': cond_cells, 'lci_cond': lci_cond,
           'lci_depth_top': lci_depth_top, 'lci_doi': lci_doi, 'line': line, 'northing': northing, 'easting': easting, 'fiducial': fiducial,
           'elevation': elevation}

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
