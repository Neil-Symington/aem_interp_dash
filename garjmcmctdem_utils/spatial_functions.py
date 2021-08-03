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

Spatial functions used for various bits an pieces
'''

from scipy.spatial.ckdtree import cKDTree
from scipy.spatial.ckdtree import cKDTree
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import math
from shapely.geometry import Point

def depth_to_thickness(depth):
    """
    Function for calculating thickness from depth array
    :param depth: an array of depths
    :return:
    a flat array of thicknesses with the last entry being a null
    """
    # Create a new thickness array
    thickness = np.nan*np.ones(shape=depth.shape,
                               dtype=np.float)
    # Iterate through the depth array
    if len(depth.shape) == 1:
        thickness[0:-1] = depth[1:] - depth[:-1]
        return thickness

    elif len(depth.shape) == 2:
        thickness[:, 0:-1] = depth[:, 1:] - depth[:, :-1]
        return thickness

    elif len(depth.shape) == 3:

        thickness[:-1,:,:] = depth[1:,:, :] - depth[:-1,:, :]
        return thickness

def thickness_to_depth(thickness):
    """
    Function for calculating depth top from a thickness array
    :param depth: an array of thicknesses
    :return:
    a flat array of depth
    """
    # Create a new thickness array
    depth = np.zeros(shape=thickness.shape,
                               dtype=np.float)
    # Iterate through the depth array
    depth[1:] = np.cumsum(thickness[:-1])

    return depth

def nearest_neighbours(points, coords, points_required = 1,max_distance = 250.):

    """
    An implementation of nearest neaighbour for spatial data that uses kdtrees

    :param points: array of points to find the nearest neighbour for
    :param coords: coordinates of points
    :param points_required: number of points to return
    :param max_distance: maximum search radius
    :return:
    """
    if len(np.array(points).shape) == 1:
        points = np.array([points])

    # Initialise tree instance
    kdtree = cKDTree(data=coords)

    # iterate throught the points and find the nearest neighbour
    distances, indices = kdtree.query(points, k=points_required,
                                      distance_upper_bound=max_distance)

    # Mask out infitnite distances in indices to avoid confusion
    mask = np.isfinite(distances)

    if not np.all(mask):

        distances[~mask] = np.nan

    return distances, indices

def  layer_centre_to_top(layer_centre_depth):
    """Function for getting layer top depth from layer centre depths. Assumes
    that depth starts at zero.

    Parameters
    ----------
    layer_centre_depth : array
        Description of parameter `layer_centre_depth`.

    Returns
    -------
    array
        Layer top depth

    """
    layer_top_depth = np.zeros(shape = layer_centre_depth.shape,
                              dtype = layer_centre_depth.dtype)
    layer_top_depth[:,1:] = layer_centre_depth[:,1:] - layer_centre_depth[:, :1]
    return layer_top_depth

def line_length(line):
    '''
    Function to return length of line
    @param line: iterable containing two two-ordinate iterables, e.g. 2 x 2 array or 2-tuple of 2-tuples

    @return length: Distance between start & end points in native units
    '''
    return math.sqrt(math.pow(line[1][0] - line[0][0], 2.0) +
                     math.pow(line[1][1] - line[0][1], 2.0))

def coords2distance(coordinate_array):
    '''
    From geophys_utils, transect_utils

    Function to calculate cumulative distance in metres from native (lon/lat) coordinates
    @param coordinate_array: Array of shape (n, 2) or iterable containing coordinate pairs

    @return distance_array: Array of shape (n) containing cumulative distances from first coord
    '''
    coord_count = coordinate_array.shape[0]
    distance_array = np.zeros((coord_count,), coordinate_array.dtype)
    cumulative_distance = 0.0
    distance_array[0] = cumulative_distance
    last_point = coordinate_array[0]

    for coord_index in range(1, coord_count):
        point = coordinate_array[coord_index]
        distance = line_length((point, last_point))
        cumulative_distance += distance
        distance_array[coord_index] = cumulative_distance
        last_point = point

    return distance_array

def interpolate_1d_vars(vars_1D, var_dict, resampling_method='linear'):
    """
    Interpolate the 1D variables onto regular distance axes

    """
    # Iterate through the 1D variables, interpolate them onto the distances that were used for
    # the 2D variable gridding and add it to the dictionary

    for var in vars_1D:

        varray = griddata(var_dict['distances'],
                          var_dict[var], var_dict['grid_distances'],
                          method=resampling_method)


        yield varray

def interpolate_2d_vars(vars_2d, var_dict, xres, yres):
    """
    Generator to interpolate 2d variables (i.e conductivity, uncertainty)

    :param vars_2d:
    :param var_dict:
    :param xres:
    :param yres:
    :return:
    """

    ndepth_cells = var_dict['ndepth_cells']

    # Find the depth variable

    layer_thicknesses = depth_to_thickness(var_dict['layer_top_depth'])

    # Give the bottom layer the tickness of the second bottom layer

    layer_thicknesses[:,-1] = layer_thicknesses[:,-2]

    # Get the vertical limits

    elevations = var_dict['elevation']

    max_depth = np.max(var_dict['layer_top_depth'])

    # elevation limits
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

    assert len(grid_elevations) > ndepth_cells

    # Add to the variable dictionary

    var_dict['grid_elevations'] = grid_elevations

    var_dict['grid_distances'] = grid_distances

    # Interpolate the elevation

    f = interp1d(distances, elevations)

    max_elevation = f(grid_distances)

    # Interpolate the layer thicknesses ont our grid

    grid_thicknesses = np.nan*np.ones(shape = (grid_distances.shape[0],
                                               grid_elevations.shape[0]),
                                      dtype = layer_thicknesses.dtype)

    # Iterate through each one of the layers and iterpolate
    # Nb if all layers are equally thick then this block does nothing
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
                                   ndepth_cells))

        if var == 'conductivity':

            point_var = np.log10(point_var)

        for j in range(point_var.shape[1]):

            f = interp1d(distances, point_var[:,j])
            new_var[:, j] = f(grid_distances)

        if var.startswith('conductivity'):

            new_var = 10**(new_var)

        # Now we need to place the 2d variables on the new grid
        for i in range(grid_distances.shape[0]):
            dtop = 0.
            for j in range(ndepth_cells - 1):
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

        #if var_dict['reverse_line']:
        #    interpolated_var = np.flipud(interpolated_var)

        # We also want to transpose the grid so the up elevations are up

        interpolated_var = interpolated_var.T

        # Yield the generator and the dictionary with added variables
        yield interpolated_var, var_dict
def interpolate_data(data_variables, var_dict, interpolated_utm,
                     resampling_method='linear'):
    """
    :param data_variables: variables from netCDF4 dataset to interpolate
    :param var_dict: dictionary with the arrays for each variable
    :param interpolated_utm: utm corrdinates onto which to interpolate the line data
    :param resampling_method:
    :return:
    """

    # Define coordinates
    utm_coordinates = var_dict['utm_coordinates']

    # Add distance array to dictionary
    distances = coords2distance(utm_coordinates)

    # Now we want to find the equivalent line distance of the data based on the
    # gridded coordinates

    interpolated_distances = griddata(utm_coordinates, distances, interpolated_utm,
                                      method=resampling_method)

    # Now extract the data variable, interpolate them and add them to the dictionary

    for var in data_variables:

        # Create an empty array for interpolation

        arr = var_dict[var]

        interp_arr = np.zeros(shape=(np.shape(interpolated_distances)[0], np.shape(arr)[1]),
                              dtype=var_dict[var].dtype)

        # Interpolate each column separately

        for j in range(interp_arr.shape[1]):

            vals = np.log10(arr[:, j])

            interp_arr[:, j] = 10**griddata(distances, vals, interpolated_distances,
                                            method=resampling_method)

        # Add to the dictionary

        yield interp_arr

def xy_2_var(xarray, xy, var, max_distance = 100.):
    """
    Function for finding a variable for gridded AEM sections
    given an input easting and northing
    @ param: xarray : for gridded line data
    @ param: xy: numpy array with easting and northing
    @ param: var: string with variable name
    returns
    float: distance along line
    """
    utm_coords = np.column_stack((xarray.easting.values,
                                  xarray.northing.values))

    d, i = nearest_neighbours(xy, utm_coords, max_distance=max_distance)
    if np.isnan(d).all():
        return None

    else:
        return xarray[var][i]

def return_valid_points(points, coords, extent):
    # Now get points that are within our survey area
    mask = [Point(coords[id]).within(extent) for id in points]
    u, indices = np.unique(np.array(points)[mask], return_index = True)

    return u[indices]

def interp2scatter(surface, line, gridded_data, easting_col = 'X',
                   northing_col = 'Y', elevation_col = 'ELEVATION',
                   line_col = 'SURVEY_LINE'):
    """Function for taking .

    Parameters
    ----------
    surface : instance of modelling class
        From modelling_utils.
    line : int
        line number.
    gridded_data : dictionary
        dictionary of section grids.

    Returns
    -------
    grid_dists
        Array of grid distances.
    elevs
        Array of elevations
    fids
        Array of fiducials

    """
    mask = surface.interpreted_points[line_col] == line
    utm_coords = np.column_stack((gridded_data[line]['easting'],
                                  gridded_data[line]['northing']))

    dist, inds = nearest_neighbours(surface.interpreted_points[mask][[easting_col,northing_col]].values,
                                                      utm_coords, max_distance=100.)

    grid_dists = gridded_data[line]['grid_distances'][inds]
    elevs = surface.interpreted_points[mask][elevation_col].values
    fids = surface.interpreted_points[mask].index
    return grid_dists, elevs, fids

def sort_variables(var_dict):
    """Function for sorting a dictionary of variables by fiducial then easting.
    Assumes 'easting' and fiducial are in dictionary

    Parameters
    ----------
    var_dict : dicitonary
       Dictionary of variables.

    Returns
    -------
    dictionary
       dictionary of sorted array

    """
    # First sort on fiducial
    sort_mask = np.argsort(var_dict['fiducial'])
    # Now sort from east to west if need be
    if var_dict['easting'][sort_mask][0] > var_dict['easting'][sort_mask][-1]:
       sort_mask = sort_mask[::-1]
    # now apply the mask to every variable
    for item in var_dict:
       var_dict[item] = var_dict[item][sort_mask]
    return var_dict

def scale_distance_along_line(xarr1, xarr2):
    """
    Function for scaling one xarray onto the -axis of another
    Parameters
    ----------
    xr xarray
        dataset onto which to scale grid distances
    xarr2 xarray
        dataset to scale

    Returns
    -------

    """
    # create the interpolator
    coords = np.column_stack((xarr1['easting'].values,
                              xarr1['northing'].values))

    # Now interpolate
    new_coords = np.column_stack((xarr2['easting'].values,
                                  xarr2['northing'].values))

    # Our new coordinates are always sitting between two points
    d, i = nearest_neighbours(new_coords, coords, points_required = 2,max_distance = 250.)

    weights = 1/d

    # Normalise the array so the rows sum to unit

    weights /= np.sum(weights, axis=1)[:, None]

    # get our grid distances
    return np.sum(xarr1['grid_distances'].values[i] * weights, axis=1)