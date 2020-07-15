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
Created on 16/1/2019
@author: Neil Symington

These are functions used to visualise hydrogeological data
'''

import matplotlib.pyplot as plt
from netcdf_utils import get_lines
import spatial_functions
import numpy as np
import h5py
import gc, os
import netCDF4
import math
import spatial_functions
from scipy.interpolate import griddata
from scipy.interpolate import interp1d

def AEM_baseplot(stoch_inv, det_inv, layer_number = 1):
    ## TODO add a plot parameter file
    """Create the fig and axis for a base plot showing LCI grids and rj scatter
    points.

    Parameters
    ----------
    stoch_inv : object
        Inversion class of type stochastic.
    deth_inv : object
        Inversion class of type deterministic.
    layer_number: int
        The layer number that will be plotted
    Returns
    -------
    type
        Matplotlib fig and axis.

    """
    # Do some checks
    assert hasattr(det_inv, 'layer_grids')
    assert det_inv.inversion_type == 'deterministic'
    assert stoch_inv.inversion_type == 'stochastic'

    fig, ax = plt.subplots(1,1,figsize = (12,12))

    grid = ax.imshow(np.log10(det_inv.layer_grids['Layer_' + str(layer_number)]['conductivity']),
              extent = det_inv.layer_grids['bounds'], cmap = 'jet',
              vmin = -3, vmax = 0)
    fig.colorbar(grid)

    ax.scatter(stoch_inv.coords[:,0],stoch_inv.coords[:,1], c='k', s=4.)

    ax.set_xlim(stoch_inv.conductivity_data.geospatial_east_min - 500,
                stoch_inv.conductivity_data.geospatial_east_max + 500)
    ax.set_ylim(stoch_inv.conductivity_data.geospatial_north_min - 500,
                stoch_inv.conductivity_data.geospatial_north_max + 500)
    return fig, ax


def purge_invalid_elevations(var_grid, grid_y, min_elevation_grid, max_elevation_grid, yres):
    """
    Function for purging interpolated values that sit above the maximum or below the minimum elevation
    :param var_grid:
    :param grid_y:
    :param min_elevation_grid:
    :param max_elevation_grid:
    :param yres:
    :return:
    """
    # Iterate through the
    for x_index in range(var_grid.shape[1]):
        # Get indices which are below the minimum elevation
        min_elevation_indices = np.where(grid_y[:,x_index] < min_elevation_grid[x_index] + yres)[0]

        try:
            var_grid[min_elevation_indices, x_index] = np.NaN
        except:
            pass
        # Get indices which are above the maximum elevation
        max_elevation_indices = np.where(grid_y[:,x_index] > max_elevation_grid[x_index] - yres)[0]

        try:
            var_grid[max_elevation_indices, x_index] = np.NaN
        except:
            pass

    return var_grid


def interpolate_2d_vars(vars_2d, var_dict, xres, yres):
    """
    Generator to interpolate 2d variables (i.e conductivity, uncertainty)

    :param vars_2d:
    :param var_dict:
    :param xres:
    :param yres:
    :return:
    """

    nlayers = var_dict['nlayers']

    # Get the thickness of the layers

    layer_thicknesses = spatial_functions.depth_to_thickness(var_dict['layer_top_depth'])

    # Give the bottom layer a thickness of 20 metres

    layer_thicknesses[:,-1] = 20.

    # Get the vertical limits, note guard against dummy values > 800m

    elevations = var_dict['elevation']

    # Guard against dummy values which are deeper than 900 metres

    max_depth = np.max(var_dict['layer_top_depth'][var_dict['layer_top_depth'] < 900.])

    vlimits = [np.min(elevations) - max_depth,
               np.max(elevations) + 5]

    # Get the horizontal limits

    distances = var_dict['distances']

    hlimits = [np.min(distances), np.max(distances)]

    # Get the x and y dimension coordinates

    xres = np.float(xres)
    yres = np.float(yres)

    grid_y, grid_x = np.mgrid[vlimits[1]:vlimits[0]:-yres,
                     hlimits[0]:hlimits[1]:xres]

    grid_distances = grid_x[0]

    grid_elevations = grid_y[:, 0]

    # Add to the variable dictionary

    var_dict['grid_elevations'] = grid_elevations

    var_dict['grid_distances'] = grid_distances

    # Interpolate the elevation

    f = interp1d(distances, elevations)

    max_elevation = f(grid_distances)

    # Interpolate the layer thicknesses

    grid_thicknesses = np.nan*np.ones(shape = (grid_distances.shape[0],
                                               grid_elevations.shape[0]),
                                      dtype = layer_thicknesses.dtype)

    for j in range(layer_thicknesses.shape[1]):
        # Guard against nans

        if not np.isnan(layer_thicknesses[:,j]).any():
            # Grid in log10 space
            layer_thickness = np.log10(layer_thicknesses[:, j])
            f = interp1d(distances, layer_thickness)
            grid_thicknesses[:,j] = f(grid_distances)

    # Tranform back to linear space
    grid_thicknesses = 10**grid_thicknesses

    # Interpolate the variables

    # Iterate through variables and interpolate onto new grid
    for var in vars_2d:

        interpolated_var = np.nan*np.ones(grid_thicknesses.shape,
                                          dtype = var_dict[var].dtype)

        # For conductivity we interpolate in log10 space

        point_var = var_dict[var]

        new_var = np.ones(shape = (len(grid_distances),
                                   nlayers))

        if var == 'conductivity':

            point_var = np.log10(point_var)

        for j in range(point_var.shape[1]):

            f = interp1d(distances, point_var[:,j])
            new_var[:, j] = f(grid_distances)

        if var == 'conductivity':

            new_var = 10**(new_var)

        # Now we need to place the 2d variables on the new grid
        for i in range(grid_distances.shape[0]):
            dtop = 0.
            for j in range(nlayers - 1):
                # Get the thickness
                thick = grid_thicknesses[i,j]
                # Find the elevation top and bottom
                etop = max_elevation[i] - dtop
                ebot = etop - thick
                # Get the indices for this elevation range
                j_ind = np.where((etop >= grid_elevations) & (ebot <= grid_elevations))
                # Populate the section
                interpolated_var[i, j_ind] = new_var[i,j]
                # Update the depth top
                dtop += thick

        # Reverse the grid if it is west to east

        if var_dict['reverse_line']:

            interpolated_var = np.flipud(interpolated_var)

        # We also want to transpose the grid so the up elevations are up

        interpolated_var = interpolated_var.T

        # Yield the generator and the dictionary with added variables
        yield interpolated_var, var_dict


# Pull data from h5py object to a dictionary
def extract_hdf5_grids(f, plot_vars):
    """

    :param f: hdf5 file
    :param plot_vars:
    :return:
    dictionary with interpolated datasets
    """

    datasets = {}

    for item in f.values():
        if item.name[1:] in plot_vars:
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
        if item.name[1:] == 'flm_layer_top_depth':
            datasets['flm_layer_top_depth'] = item[()]

    return datasets
