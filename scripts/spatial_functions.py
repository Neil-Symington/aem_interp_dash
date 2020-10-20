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
        distance = math.sqrt(math.pow(point[0] - last_point[0], 2.0) + math.pow(point[1] - last_point[1], 2.0))
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

        # Reverse the grid if it is west to east

        #if var_dict['reverse_line']:
        #    varray = varray[::-1]

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

def xy_2_var(grid_dict, xy, var):
    """
    Function for finding a variable for gridded AEM sections
    given an input easting and northing
    @ param: grid_dict :dictionary for gridded line data
    @ param: xy: numpy array with easting and northing
    @ param: var: string with variable name
    returns
    float: distance along line
    """
    utm_coords = np.column_stack((grid_dict['easting'],
                                  grid_dict['northing']))

    d, i = nearest_neighbours(xy, utm_coords, max_distance=100.)
    if np.isnan(d[0]):
        return None

    else:
        near_ind = i[0]
        return grid_dict[var][near_ind]

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
