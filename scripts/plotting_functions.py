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

def AEM_baseplot(stoch_inv, det_inv, layer_number = 1, plot_args = {}):
    ## TODO add a plot parameter file
    """Create the fig and axis for a base plot showing LCI grids and rj scatter
    points.

    Parameters
    ----------
    stoch_inv : object
        Inversion class of type stochastic.
    deth_inv : object
        Inversion class of type deterministic.
    plot_args: dictionary
        Dictionary with plotting variables
    Returns
    -------
    type
        Matplotlib fig and axis.

    """
    # custom plot vars
    custom_args = {'Layer_number': 10, "vmin": 0.001, "vmax": 1,
                   'colour_stretch': 'jet',
                   "point_size": 4, "figsize": (12,12), "point_colour": 'k',
                   'buffer': 500.}


    # Do some checks
    assert hasattr(det_inv, 'layer_grids')
    assert det_inv.inversion_type == 'deterministic'
    assert stoch_inv.inversion_type == 'stochastic'

    for item in custom_args.keys():
        if item not in plot_args.keys():
            plot_args[item] = custom_args


    fig, ax = plt.subplots(1,1,figsize = (12,12))

    layer_number = str(int(plot_args['Layer_number']))

    cond_grid = np.log10(det_inv.layer_grids['Layer_' + layer_number]['conductivity'])

    im = ax.imshow(cond_grid, extent = det_inv.layer_grids['bounds'],
                     cmap = plot_args['colour_stretch'],
                     vmin = np.log10(plot_args['vmin']),
                     vmax =np.log10(plot_args['vmax']))

    ax.scatter(stoch_inv.coords[:,0],stoch_inv.coords[:,1], c=plot_args['point_colour'],
               s = plot_args['point_size'])

    buffer = plot_args['buffer']
    ax.set_xlim(stoch_inv.data.geospatial_east_min - buffer,
                stoch_inv.data.geospatial_east_max + buffer)
    ax.set_ylim(stoch_inv.data.geospatial_north_min - buffer,
                stoch_inv.data.geospatial_north_max + buffer)

    # Add tick axis
    cax = fig.add_axes([0.9, 0.25, 0.02, 0.6])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.set_yticklabels([round(10 ** x, 4) for x in cb.get_ticks()])

    return fig, ax, cax

def interpreted_surface_dual_plot(surface,
                                 plot_args = {'Panel_1':{}, 'Panel_2': {}},
                                 update_grid = True):
    """Create a two panel plot showing the interpolated grids and interpreted
    points. For examples this could be comparing layer_elevation and layer_depth.
    Or it could be showing layer elevation and layer uncertainty.

    Parameters
    ----------
    surface : object
        An instance of modelled_oundary class
    plot_args: dictionary
        Dictionary with plotting arguments

    Returns
    -------
    figure and axis objects
        Description of returned object.

    """
    # custom plot vars
    custom_args = {'Panel_1': {'variable': 'layer_elevation_grid',
                         'grid': 'standard_deviation_grid',
                         'interpolator': 'standard_deviation_gp',
                          "vmin": -300., "vmax": 0.,
                         'colour_stretch': 'viridis'},

            'Panel_2': {'variable': 'layer_elevation_grid',
                         'grid': 'layer_elevation_grid',
                         'interpolator': 'layer_elevation_gp',
                        "vmin": 0., "vmax": 50.,
                        'colour_stretch': 'viridis'}}

    # For ease of use
    grid_names = [plot_args["Panel_1"]['grid'], plot_args["Panel_2"]['grid']]
    interpolator_names = [plot_args["Panel_1"]['interpolator'], plot_args["Panel_2"]['interpolator']]
    var_names = [plot_args["Panel_1"]['variable'], plot_args["Panel_2"]['variable']]
    # Do some checks
    for i in range(2):
        assert hasattr(surface, grid_names[i])
        assert hasattr(surface, interpolator_names[i])
        assert var_names[i] in surface.interpreted_points.keys()
    # If plot arguments are not given we use the custom
    for panel in custom_args.keys():
        for item in custom_args[panel].keys():
            if item not in plot_args[panel].keys():
                plot_args[panel][item] = custom_args

    if update_grid:
        for i in range(2):
            surface.fit_interpolator(variable = var_names[i],
                                     interpolator_name = interpolator_names[i])

            surface.predict_on_grid(interpolator_name = interpolator_names[i],
                                    grid_name = grid_names[i])

    fig, ax_array = plt.subplots(1,2, figsize = (12,6))

    extent = surface.bounds

    grid_1 = getattr(surface, grid_names[0])

    im1 = ax_array[0].imshow(grid_1,extent = extent,
                     vmin = plot_args['Panel_1']['vmin'],
                     vmax = plot_args['Panel_1']['vmax'],
                     cmap = plot_args['Panel_1']['colour_stretch'])

    cax = fig.add_axes([0.4, 0.6, 0.01, 0.25])

    fig.colorbar(im1, cax=cax)

    grid_2 = getattr(surface, grid_names[1])

    im2 = ax_array[1].imshow(grid_2,extent = extent,
                     vmin = plot_args['Panel_2']['vmin'],
                     vmax = plot_args['Panel_2']['vmax'],
                     cmap = plot_args['Panel_2']['colour_stretch'])

    cax2 = fig.add_axes([0.85, 0.6, 0.01, 0.25])

    fig.colorbar(im2, cax=cax2)


    X = np.column_stack((surface.interpreted_points['easting'],
                         surface.interpreted_points['northing']))


    ax_array[0].scatter(X[:,0], X[:,1], marker = 'o',
                c = surface.interpreted_points[var_names[0]],
                vmin = plot_args['Panel_1']['vmin'],
                vmax = plot_args['Panel_1']['vmax'],
                edgecolors  = 'k')

    ax_array[1].scatter(X[:,0], X[:,1], marker = 'o',
                c = surface.interpreted_points[var_names[1]],
                vmin = plot_args['Panel_2']['vmin'],
                vmax = plot_args['Panel_2']['vmax'],
                edgecolors  = 'k')

    ax_array[0].set_title(var_names[0])
    ax_array[1].set_title(var_names[1])

    return fig, ax_array, [cax, cax2]

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
