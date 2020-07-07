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
import gc

class ConductivitySections:
    """
    VerticalSectionPlot class for functions for creating vertical section plots
    from netcdf file
    """

    def __init__(self,netCDFConductivityDataset = None, netCDFemDataset = None):

        """
        :param netCDFConductivityDataset: netcdf line dataset with
         conductivity model
        :param netCDFemDataset: netcdf line dataset with
         EM measurements
        """
        if netCDFConductivityDataset is not None:
            if not self.testNetCDFDataset(netCDFConductivityDataset):
                raise ValueError("Input datafile is not netCDF4 format")
            else:
                self.conductivity_model = netCDFConductivityDataset
                self.conductivity_variables = []
        else:
            self.conductivity_model = None

        # If datafile is given then check it is a netcdf file

        if netCDFemDataset is not None:
            if not self.testNetCDFDataset(netCDFemDataset):
                raise ValueError("Input datafile is not netCDF4 format")
            else:
                self.EM_data = netCDFemDataset
                self.dataLineUtils = NetCDFLineUtils(self.EM_data)
                self.EM_variables = []
        else:
            self.EM_data = None

    def save_dict_to_hdf5(self, fname, dictionary):
        """
        Save a dictionary to hdf5
        """
        f = h5py.File(fname, "w")

        for key in dictionary.keys():
            dset = f.create_dataset(key, data=dictionary[key])
        f.close()

    def testNetCDFDataset(self, netCDF_dataset):
        """
        A  function to test if correctly if file is formatted netCDF4 file

        :param netCDF_dataset: netCDF4 dataset
        :return:

        True if correct, False if not
        """

        return netCDF_dataset.__class__ == netCDF4._netCDF4.Dataset

    def interpolate_data_coordinates(self, line, var_dict, gridding_params):
        """

        :param line:
        :param var_dict:
        :param gridding_params:
        :return:
        """
        # Create a dictionary into whcih to write interpolated coordinates
        interpolated = {}

        # Define coordinates
        utm_coordinates = np.columns_stack(var_dict['easting'],
                                           var_dict['northing'])

        if utm_coordinates[0, 0] > utm_coordinates[-1, 0]:
            var_dict['reverse_line'] = True
        else:
            var_dict['reverse_line'] = False

        # Find distance along the line
        distances = coords2distance(utm_coordinates)
        var_dict['distances'] = distances

        # Calculate 'grid' distances

        var_dict['grid_distances'] = np.arange(distances[0], distances[-1], gridding_params['xres'])

        # Interpolate the two coordinate variables
        interp1d = interpolate_1d_vars(['easting', 'northing'],
                                       var_dict, gridding_params['resampling_method'])

        for var in ['easting', 'northing']:
            # Generator yields the interpolated variable array
            interpolated[var] = next(interp1d)

        return interpolated, var_dict


    def grid_conductivity_variables(self, line, cond_var_dict, gridding_params, smoothed = False):

        """

        :param line:
        :param cond_var_dict:
        :return:
        """

        # Create an empty dictionary
        interpolated = {}

        # If the line is west to east we want to reverse the coord
        # array and flag it

        # Define coordinates
        utm_coordinates = np.column_stack((cond_var_dict['easting'],
                                          cond_var_dict['northing']))


        # Add the flag to the dictionary
        if utm_coordinates[0, 0] > utm_coordinates[-1, 0]:
            cond_var_dict['reverse_line'] = True
        else:
            cond_var_dict['reverse_line'] = False

        # Add distance array to dictionary
        cond_var_dict['distances'] = coords2distance(utm_coordinates)

        # Add number of layers to the array
        cond_var_dict['nlayers'] = self.conductivity_model.dimensions['layer'].size

        # Interpolate 2D and 1D variables

        vars_2d = [v for v in self.conductivity_variables if cond_var_dict[v].ndim == 2]
        vars_1d = [v for v in self.conductivity_variables if cond_var_dict[v].ndim == 1]

        # Generator for inteprolating 2D variables from the vars_2d list
        if not smoothed:
            interp2d = interpolate_2d_vars(vars_2d, cond_var_dict, gridding_params['xres'],
                                       gridding_params['yres'])
        else:
            interp2d = interpolate_2d_vars_smooth(vars_2d, cond_var_dict, gridding_params['xres'],
                                           gridding_params['yres'], gridding_params['layer_subdivisions'],
                                           gridding_params['resampling_method'])

        for var in vars_2d:
            # Generator yields the interpolated variable array
            interpolated[var], cond_var_dict = next(interp2d)

        # Add grid distances and elevations to the interpolated dictionary
        interpolated['grid_distances'] = cond_var_dict['grid_distances']
        interpolated['grid_elevations'] = cond_var_dict['grid_elevations']

        # Generator for inteprolating 1D variables from the vars_1d list
        interp1d = interpolate_1d_vars(vars_1d, cond_var_dict,
                                       gridding_params['resampling_method'])

        for var in vars_1d:
            # Generator yields the interpolated variable array
            interpolated[var] = next(interp1d)

        return interpolated

    def xy_2_var(self, grid_dict, xy, var):
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

        d, i = spatial_functions.nearest_neighbours(xy,
                                                    utm_coords,
                                                    points_required=1,
                                                    max_distance=100.)
        if np.isnan(d[0]):
            return None

        else:
            near_ind = i[0]

            return grid_dict[var][near_ind]

    def grid_variables(self, xres, yres, lines,
                       layer_subdivisions = None, resampling_method = 'linear',
                       smoothed = False, save_hdf5 = False, hdf5_dir = None,
                       overwrite_hdf5 = True, return_dict = True):
        """
        A function for interpolating 1D and 2d variables onto a vertical grid
        cells size xres, yres
        :param xres: Float horizontal cell size along the line
        :param yres: Float vertical cell size
        :param lines: int single line or list of lines to be gridded
        :param layer_subdivisions:
        :param resampling_method: str or int, optional - from scipy gridata
        :param save_hdf5: Boolean parameter indicating whether interpolated variables
         get saved as hdf or no
        :param hdf5_dir: path of directory into which the hdf5 files are saved
        :param overwrite_hdf5: Boolean parameter referring to if the user wants to
         overwrite any pre-existing files
        :param return_dict: Boolean parameter indicating if a dictionary is returned or not
        :return:
        dictionary with interpolated variables as numpy arrays
        """

        # Create a line utils for each object if the objects exist
        if self.conductivity_model is not None:
            # Flag for if dta was included in the plot section initialisation
            plot_cond = True
            # Add key variables if they aren't in the list to grid
            for item in ['easting', 'northing', 'elevation', 'layer_top_depth']:
                if item not in self.conductivity_variables:
                    self.conductivity_variables.append(item)
        else:
            plot_cond = False

        if self.EM_data is not None:
            # Flag for if dta was included in the plot section initialisation
            plot_dat = True

        else:
            plot_dat = False

        # If line is not in an array like object then put it in a list
        if type(lines) == int:
            lines = [lines]
        elif isinstance(lines ,(list, tuple, np.ndarray)):
            pass
        else:
            raise ValueError("Check lines variable.")

        # First create generators for returning coordinates and variables for the lines

        if plot_cond:
            cond_lines= get_lines(self.conductivity_model,
                                  line_numbers=lines,
                                  variables=self.conductivity_variables)
        if plot_dat:
            dat_lines = get_lines(self.EM_data, line_numbers=lines,
                                  variables=self.EM_variables)

        # Interpolated results will be added to a dictionary
        interpolated = {}

        # Create a gridding parameters dictionary

        gridding_params = {'xres': xres, 'yres': yres,
                          'layer_subdivisions': layer_subdivisions,
                           'resampling_method': resampling_method}

        # Iterate through the lines
        for i in range(len(lines)):

            # Extract the variables and coordinates for the line in question
            if plot_cond:

                line_no, cond_var_dict = next(cond_lines)

                cond_var_dict['utm_coordinates'] = np.column_stack((cond_var_dict['easting'],
                                                                    cond_var_dict['northing']))

                interpolated[line_no] =  self.grid_conductivity_variables(line_no, cond_var_dict,
                                                                          gridding_params, smoothed=smoothed)

            if plot_dat:
                # Extract variables from the data
                line_no, data_var_dict = next(dat_lines)

                data_var_dict['utm_coordinates'] = np.column_stack((cond_var_dict['easting'],
                                                                    cond_var_dict['northing']))

                # If the conductivity variables have not been plotted then we need to interpolate the coordinates

                if not plot_cond:

                    interpolated[line_no], data_var_dict = self.interpolate_data_coordinates(line_no,data_var_dict,
                                                                                        gridding_params)

                interpolated_utm = np.column_stack((interpolated[line_no]['easting'],
                                              interpolated[line_no]['northing']))

                # Generator for interpolating data variables from the data variables list
                interp_dat = interpolate_data(self.EM_variables, data_var_dict, interpolated_utm,
                                              resampling_method)

                for var in self.EM_variables:

                    interpolated[line_no][var] = next(interp_dat)

            # Save to hdf5 file if the keyword is passed
            if save_hdf5:
                fname = os.path.join(hdf5_dir, str(line_no) + '.hdf5')
                if overwrite_hdf5:
                    self.save_dict_to_hdf5(fname, interpolated[line_no])
                else:
                    if os.path.exists(fname):
                        print("File ", fname, " already exists")
                    else:
                        self.save_dict_to_hdf5(fname, interpolated[line_no])

            # Many lines may fill up memory so if the dictionary is not being returned then
            # we garbage collect
            if not return_dict:

                del interpolated[line_no]

                # Collect the garbage
                gc.collect()

        if return_dict:
            return interpolated

def save_dict_to_hdf5(fname, dictionary):
    """
    Save a dictionary to hdf5
    """
    f = h5py.File(fname, "w")
    for key in dictionary.keys():
        dset = f.create_dataset(key, data=dictionary[key])
    f.close()

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
