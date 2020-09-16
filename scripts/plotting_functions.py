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
from matplotlib.colors import Normalize
import matplotlib as mpl

def AEM_baseplot(stoch_inv, det_inv, layer_number = 1, plot_args = {}):
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

    layer = det_inv.layer_grids['Layer_{}'.format(plot_args['Layer_number'])]

    fig, ax = plt.subplots(1,1,figsize = plot_args['figsize'])

    layer_number = str(int(plot_args['Layer_number']))

    cond_grid = np.log10(layer['conductivity'])

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
    cax = fig.add_axes([0.92, 0.25, 0.01, 0.6])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.set_yticklabels([round(10 ** x, 4) for x in cb.get_ticks()])
    cb.ax.tick_params(labelsize=8)

    depth_from = layer['depth_from']
    depth_to = layer['depth_to']

    ax.set_title('Conductivity grid for {} to {} mBGL'.format(depth_from, depth_to))

    return fig, ax, cax

def interpreted_surface_dual_plot(surface,plot_args = {'Panel_1':{}, 'Panel_2': {}},update_grid = True):

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
                         'colour_stretch': 'magma'},
                  'fig_args': {'figsize': (10,10)}}

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
    for key in custom_args.keys():

        for item in custom_args[key].keys():
            if item not in plot_args[key].keys():
                plot_args[key][item] = custom_args


    if update_grid:
        for i in range(2):
            surface.fit_interpolator(variable = var_names[i],
                                     interpolator_name = interpolator_names[i])

            surface.predict_on_grid(interpolator_name = interpolator_names[i],
                                    grid_name = grid_names[i])

    fig, ax_array = plt.subplots(1,2, figsize = plot_args['fig_args']['figsize'])

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
                edgecolors  = 'k',
                cmap = plot_args['Panel_1']['colour_stretch'])

    ax_array[1].scatter(X[:,0], X[:,1], marker = 'o',
                c = surface.interpreted_points[var_names[1]],
                vmin = plot_args['Panel_2']['vmin'],
                vmax = plot_args['Panel_2']['vmax'],
                edgecolors  = 'k',
                cmap = plot_args['Panel_2']['colour_stretch'])

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

def plot_grid(ax, gridded_variables, variable, panel_kwargs, x_ax_var='grid_distances'):
    """Short summary.

    Parameters
    ----------
    ax : type
        Description of parameter `ax`.
    gridded_variables : type
        Description of parameter `gridded_variables`.
    variable : type
        Description of parameter `variable`.
    panel_kwargs : type
        Description of parameter `panel_kwargs`.
    x_ax_var : type
        Description of parameter `x_ax_var`.

    Returns
    -------
    type
        Description of returned object.

    """


    # Define extents based on kwarg max depth

    try:
        min_elevation = np.min(gridded_variables['elevation']) - panel_kwargs['max_depth']

    except KeyError:

        min_elevation = gridded_variables['grid_elevations'][-1]

    extent = (gridded_variables[x_ax_var][0], gridded_variables[x_ax_var][-1],
              gridded_variables['grid_elevations'][-1], gridded_variables['grid_elevations'][0])

    # WE will make the ylim 10% of the depth range

    max_elevation = gridded_variables['grid_elevations'][0] + 0.1 * (gridded_variables['grid_elevations'][0]
                                                                     - min_elevation)

    ax.set_ylim(min_elevation, max_elevation)

    # Define stretch
    # Flag for a logarithmic stretch

    try:
        log_stretch = panel_kwargs['log_plot']

    except KeyError:
        log_stretch = True  # False unless otherwise specified

    if log_stretch:
        # Tranform the plot data
        data = np.log10(gridded_variables[variable])

    else:
        data = gridded_variables[variable]
        # set automatic stretch values in case vmin and vmax aren't specified
        vmin, vmax = 0, 0.5

    # Define vmin an vmax if specified
    if 'vmin' in panel_kwargs.keys():
        vmin = panel_kwargs['vmin']
    if 'vmax' in panel_kwargs.keys():
        vmax = panel_kwargs['vmax']

    if log_stretch:
        vmin, vmax = np.log10(vmin), np.log10(vmax)

    # Define cmap if it is specified
    if 'cmap' in panel_kwargs.keys():
        cmap = panel_kwargs['cmap']

    else:
        cmap = 'jet'

    # Plot data

    im = ax.imshow(data, vmin=vmin, vmax=vmax,
                   extent=extent,
                   aspect='auto',
                   cmap=cmap)

    # Plot the elevation as a line over the section
    line_x = np.linspace(gridded_variables[x_ax_var][0], gridded_variables[x_ax_var][-1],
                         np.shape(gridded_variables[variable])[1])

    ax.plot(line_x, gridded_variables['elevation'], 'k')

    # To remove gridded values that stick above this line we will fill the sky in as white
    ax.fill_between(line_x, max_elevation * np.ones(np.shape(line_x)),
                    gridded_variables['elevation'], interpolate=True, color='white', alpha=1)

    # Add ylabel
    try:
        ylabel = panel_kwargs['ylabel']
        ax.set_ylabel(ylabel)
    except KeyError:
        pass

    # PLot depth of investigation and make area underneath more transparent if desired
    if panel_kwargs['shade_doi']:
        eoi = gridded_variables['elevation'] - gridded_variables['depth_of_investigation']

        ax.plot(line_x, eoi, 'k')

        grid_base = gridded_variables['grid_elevations'][-1]

        # Shade the belwo doi areas

        ax.fill_between(line_x, eoi, grid_base, interpolate=True, color='white', alpha=0.5)

    return im

def plot_single_line(ax, gridded_variables, variable, panel_kwargs,  x_ax_var='grid_distances'):

    """

    :param ax:
    :param gridded_variables:
    :param variables:
    :param panel_kwargs:
    :return:
    """
    # Define the array

    data = gridded_variables[variable]

    if 'colour' in panel_kwargs.keys():
        colour = panel_kwargs['colour']
    else:
        colour = 'black'

    lin = ax.plot(gridded_variables[x_ax_var], data, colour)

    # Extract ymin and ymax if specified, otherwise assign based on the range with the line dataset
    if 'ymin' in panel_kwargs.keys():
        ymin = panel_kwargs['ymin']
    else:
        ymin = np.min(data) - 0.1 * np.min(data)

    if 'ymax' in panel_kwargs.keys():
        ymax = panel_kwargs['ymax']
    else:
        ymax = np.max(data) - 0.1 * np.max(data)

    ax.set_ylim(bottom=ymin, top=ymax, auto=False)

    try:
        ylabel = panel_kwargs['ylabel']
        ax.set_ylabel(ylabel)
    except KeyError:
        pass

    try:
        if panel_kwargs['legend']:
            ax.legend()
    except KeyError:
        pass

    return lin

def pmap_plot(D, pmap_kwargs, surface, lci, figsize = (8,8), outfile = None):

    fig = plt.figure(figsize = figsize)

    ax1 = fig.add_axes([0.05, 0.35, 0.35, 0.62])
    ax2 = fig.add_axes([0.45, 0.35, 0.2, 0.62])
    ax3 = fig.add_axes([0.70, 0.52, 0.2, 0.2])
    ax4 = fig.add_axes([0.72, 0.32, 0.16, 0.16])
    ax5 = fig.add_axes([0.1, 0.18, 0.76, 0.05])
    ax6 = fig.add_axes([0.1, 0.05, 0.76, 0.13])
    ax7 = fig.add_axes([0.70, 0.78, 0.2, 0.2])
    cbar_ax1 = fig.add_axes([0.05, 0.29, 0.35, 0.01])
    cbar_ax2 = fig.add_axes([0.88, 0.05, 0.01, 0.2])
    cbar_ax3 = fig.add_axes([0.9, 0.52, 0.01, 0.2])


    # Plot probability map

    # ax1
    im = ax1.imshow(D['conductivity_pdf'], extent = D['conductivity_extent'],
                    aspect = 'auto', cmap = pmap_kwargs['panel_1']['cmap'])

    #  PLot the median, and percentile plots
    ax1.plot(np.log10(D['cond_p10']), D['depth_cells'], c = 'k',linestyle='dashed', label = 'p10')
    ax1.plot(np.log10(D['cond_p90']), D['depth_cells'], c = 'k',linestyle='dashed', label = 'p90')
    ax1.plot(np.log10(D['cond_p50']), D['depth_cells'], c = 'k',label = 'p50')
    ax1.plot(np.log10(D['cond_mean']), D['depth_cells'], c = 'grey',label = 'mean')

    ax1.set_xticklabels([round(10 ** float(x), 4) for x in ax1.get_xticks()])

    # for lci layered model we do some processing
    lci_expanded = np.zeros(shape=2 * len(D['lci_cond']) + 1,
                                 dtype=np.float)

    lci_expanded[1:] = np.repeat(D['lci_cond'], 2)

    depth_expanded = (np.max(D['lci_depth_top']) + 10) * np.ones(shape=len(lci_expanded),
                                                            dtype=np.float)

    depth_expanded[:-1] = np.repeat(D['lci_depth_top'], 2)

    ax1.plot(np.log10(lci_expanded), depth_expanded, c = 'pink',
             linestyle = 'dashed', label = 'lci')
    ax1.plot(ax1.get_xlim(), [D['lci_doi'], D['lci_doi']], c = 'yellow',
             label = 'LCI doi')
    ax1.set_title('rj-MCMC probability map')
    ax1.set_ylabel('depth (mBGL)')
    ax1.set_xlabel('Conductivity (S/m)')
    ax1.grid(which = 'both')

    ax1.set_ylim(pmap_kwargs['panel_1']['max_depth'],
                 pmap_kwargs['panel_1']['min_depth'])

    ax1.legend(loc = 3)

    # Ax 2
    ax2.plot(D['change_point_pdf'], D['depth_cells'], label = 'P(change point)')
    ax2.set_ylim(ax2.get_ylim()[::-1])
    ax2.set_yticks(np.arange(0, 500, 20.))
    ax2.set_title('change point probability')
    ax2.set_ylim(ax1.get_ylim())

    if not pmap_kwargs['panel_2']['auto_xlim']:
        ax2.set_xlim(pmap_kwargs['panel_2']['pmin'],
                    pmap_kwargs['panel_2']['pmax'])

    ax2.legend()
    ax2.grid(which = 'both')

    if hasattr(surface, "layer_elevation_grid"):
        elevation_grid = surface.layer_elevation_grid
        extent = surface.bounds

        im3 = ax3.imshow(elevation_grid,extent = extent,
                         vmin = pmap_kwargs['panel_3']['vmin'],
                         vmax = pmap_kwargs['panel_3']['vmax'])

        ax3.scatter(surface.interpreted_points['easting'],
                    surface.interpreted_points['northing'], c='k',
                    marker = '+', s = 0.5)

        ax3.plot(D['easting'],D['northing'],  'x', c = 'red')

        cb3 =  fig.colorbar(im3, cax=cbar_ax3, orientation='vertical')
        cb3.set_label('surface elevation mAHD')

    # Ax 4
    sample = D['sample_no'][:]

    # Add the misfit
    for i in range(D['misfit'].shape[0]):

        misfits = D['misfit'][i]
        ax4.plot(sample, misfits/D['ndata'])

    ax4.plot([1, D['nsamples']], [1,1], 'k')
    ax4.plot([D['burnin'], D['burnin']],[0.01,1e4], 'k')
    #ax4.set_xlim([1, D['misfit'].shape[1]])
    ax4.set_ylim(pmap_kwargs['panel_4']['misfit_min'],
                 pmap_kwargs['panel_4']['misfit_max'])

    ax4.set_xscale('log')
    ax4.set_yscale('log')

    ax4.set_xlabel("sample #")
    ax4.set_ylabel("Normalised misfit")

    # Ax 5
    line = D['line']

    dist = D['lci_dist']

    res1 = plot_single_line(ax5, D['lci_line'],
                                 'data_residual', pmap_kwargs['panel_5'])

    ax5.set_title('LCI conductivity section - ' + str(line))

    # Ax 6

    # Find distance along the lci section


    im2 = plot_grid(ax6, D['lci_line'], 'conductivity',
                              panel_kwargs = pmap_kwargs['panel_6'])

    ax6.plot([dist, dist], [-500, 500], 'pink')
    ax6.set_xlabel("Distance along line (m)")

    ax5.set_xlim(dist - pmap_kwargs['panel_5']['buffer'],
                 dist + pmap_kwargs['panel_5']['buffer'])
    ax6.set_xlim(dist - pmap_kwargs['panel_6']['buffer'],
                 dist + pmap_kwargs['panel_6']['buffer'])

    # Ax7
    layer = pmap_kwargs['panel_7']['Layer_number']
    cond_grid = np.log10(lci.layer_grids['Layer_{}'.format(layer)]['conductivity'])

    im7 = ax7.imshow(cond_grid, extent = lci.layer_grids['bounds'],
                     cmap = pmap_kwargs['panel_7']['cmap'],
                     vmin = np.log10(pmap_kwargs['panel_7']['vmin']),
                     vmax =np.log10(pmap_kwargs['panel_7']['vmax']))

    ax7.set_xlim(D['easting'] - pmap_kwargs['panel_7']['buffer'],
                 D['easting'] + pmap_kwargs['panel_7']['buffer'])
    ax7.set_ylim(D['northing'] - pmap_kwargs['panel_7']['buffer'],
                 D['northing'] + pmap_kwargs['panel_7']['buffer'])
    ax7.plot(D['easting'],D['northing'],  'x', c = 'k')

    p1 = [lci.section_data[line]['easting'][0], lci.section_data[line]['easting'][-1]]
    p2 = [lci.section_data[line]['northing'][0], lci.section_data[line]['northing'][-1]]
    ax7.plot(p1, p2, 'k', linewidth = 0.5)
    ax7.set_title('LCI layer slice {}'.format(layer), fontsize=10)
    ax7.tick_params(axis='both', which='major', labelsize=8)
    ax7.tick_params(axis='both', which='minor', labelsize=8)

    # cbar axes
    cb1 = fig.colorbar(im, cax=cbar_ax1, orientation='horizontal')
    cb1.set_label('probabilitiy', fontsize=10)


    cb2 = fig.colorbar(im2, cax=cbar_ax2, orientation='vertical')

    cb2.ax.set_yticklabels([round(10 ** x, 4) for x in cb2.get_ticks()])
    cb2.set_label('conductivity (S/m)', fontsize=10)


    ax_array = np.array([ax1, ax2, ax3, ax4, ax5, ax6, ax7])

    return fig, ax_array

def point_selection_plot(surface, coords, plot_args = {'Panel_1':{}, 'Panel_2': {}}, update_grid = True):

    """

    """
    # custom plot vars
    custom_args = {'Panel_1': {'variable': 'layer_elevation_grid',
                               'grid': 'standard_deviation_grid',
                               'interpolator': 'standard_deviation_gp',
                                "vmin": -300., "vmax": 0.,
                                'colour_stretch': 'viridis'},

                 'Panel_2': {"vmin": 0., "vmax": 50.,
                         'colour_stretch': 'magma'},
                  'fig_args': {'figsize': (10,10)}}

    # For ease of use
    grid_names = [plot_args["Panel_1"]['grid'], plot_args["Panel_1"]['grid'] + '_std']
    interpolator_name = plot_args["Panel_1"]['interpolator']
    var_name = plot_args["Panel_1"]['variable']
    # Do some checks
    for i in range(2):
        assert hasattr(surface, grid_names[i])

    assert hasattr(surface, interpolator_name)
    assert var_name in surface.interpreted_points.keys()

    # If plot arguments are not given we use the custom
    for key in custom_args.keys():

        for item in custom_args[key].keys():
            if item not in plot_args[key].keys():
                plot_args[key][item] = custom_args


    if update_grid:
        surface.fit_interpolator(variable = var_name,
                                 interpolator_name = interpolator_name)
        surface.predict_on_grid(interpolator_name = interpolator_name,
                                    grid_name = grid_names[0])

    fig, ax_array = plt.subplots(1,2, figsize = plot_args['fig_args']['figsize'])

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
                c = surface.interpreted_points[var_name],
                vmin = plot_args['Panel_1']['vmin'],
                vmax = plot_args['Panel_1']['vmax'],
                edgecolors  = 'k',
                cmap = plot_args['Panel_1']['colour_stretch'])

    ax_array[1].scatter(X[:,0], X[:,1], marker = 'o',
                facecolors='none',
                vmin = plot_args['Panel_2']['vmin'],
                vmax = plot_args['Panel_2']['vmax'],
                edgecolors  = 'k')

    ax_array[0].scatter(coords[:,0], coords[:,1], c = 'k', marker = 'x', s = 2)
    ax_array[1].scatter(coords[:,0], coords[:,1], c = 'k', marker = 'x', s  = 2)

    ax_array[0].set_xlim(surface.bounds[:2])
    ax_array[0].set_ylim(surface.bounds[2:])
    ax_array[1].set_xlim(surface.bounds[:2])
    ax_array[1].set_ylim(surface.bounds[2:])


    return fig, ax_array, [cax, cax2]

def percentiles2pnci(p10, p90, upper_threshold = 0.9, lower_threshold = 0.1):
    """Estimate the prior-normalised credibility interval from the 10th
    and 90th percentile gridded sections. This can be used on section plots as
    an alpha value of on its own

    Parameters
    ----------
    p10 : array
        Description of parameter `p10`.
    p90 : array
        Description of parameter `p90`.
    upper_threshold : float
        Description of parameter `upper_threshold`.
    lower_threshold : float
        Description of parameter `lower_threshold`.

    Returns
    -------
    array
        pnci array with identical shape to p10, p90

    """
    deltaP = np.log10(p90) - np.log10(p10)

    #credible interval
    ci = p90-p10
    frac = ci/deltaP

    frac_max = upper_threshold * np.ones(shape = p90.shape, dtype = np.float)
    frac_min = lower_threshold * np.ones(shape = p90.shape, dtype = np.float)

    alpha = 0 + (frac_max - frac)/(frac_max - frac_min)
    # Make alpha = 1 if the frac is less than lower threshold
    alpha[frac < frac_min] = 1.
    # Make alpha 0 if frac is higher than the upper threshold
    alpha[frac > frac_max] = 0.
    return alpha

def array2rgba(section_data, line, vmin, vmax, cmap,
               upper_threshold = 0.9, lower_threshold = 0.1):
    """Function for getting the rgba .

    Parameters
    ----------
    section_data : type
        Description of parameter `dataset`.
    line : type
        Description of parameter `line`.
    vmin : type
        Description of parameter `vmin`.
    vmax : type
        Description of parameter `vmax`.
    cmap : type
        Description of parameter `cmap`.
    upper_threshold : type
        Description of parameter `upper_threshold`.
    lower_threshold : type
        Description of parameter `lower_threshold`.

    Returns
    -------
    type
        Description of returned object.

    """

    p10 = section_data[line]['conductivity_p10']
    p90 = section_data[line]['conductivity_p90']
    p50 = section_data[line]['conductivity_p50']

    vmin, vmax = np.log10(vmin), np.log10(vmax)

    rgb = Normalize(vmin,vmax)(np.log10(p50))

    alphas = percentiles2pnci(p10,p90, upper_threshold, lower_threshold)

    cmap = getattr(plt.cm, cmap)

    colours = cmap(rgb)

    colours[..., -1] = alphas

    return colours
