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
Created on 10/7/2019
@author: Neil Symington

Code for creating AEM inversion and data objects

'''
import pickle
import h5py
import numpy as np
import spatial_functions
from netcdf_utils import get_lines, testNetCDFDataset, get_lookup_mask
from misc_utils import check_list_arg, dict_to_hdf5, extract_hdf5_data
import gc, glob, os
from shapely.geometry import LineString


class AEM_inversion:
    """
    Class for handling AEM inversions
    """

    def __init__(self,name = '', inversion_type = 'deterministic', netcdf_dataset = None):
        """Initialise instance of AEM inversion class.
        Parameters
        ----------
        inversion_type : string
            One of 'deterministic' or 'stochastic'
        netcdf_dataset : type
            netcdf dataset contain AEM inversion data
        Returns
        -------
        type
            Description of returned object.

        """
        self.name = name
        # Check inversion type
        if inversion_type in ['deterministic', 'stochastic']:
            self.inversion_type = inversion_type
        else:
            raise ValueError("inversion_type must be either deterministic or stochastic")
        # Check netcdf file
        if netcdf_dataset is not None:
            if testNetCDFDataset(netcdf_dataset):
                self.data = netcdf_dataset
            else:
                raise ValueError("Input datafile is not netCDF4 format")
                return None
            # Create an instance variable of the coordinates
            self.coords = np.column_stack((netcdf_dataset['easting'][:],
                                           netcdf_dataset['northing'][:])).data
            # Get some of the usefule metadata from the netcdf file
            self.xmin = np.min(netcdf_dataset['easting'][:])
            self.xmax = np.max(netcdf_dataset['easting'][:])
            self.ymin = np.min(netcdf_dataset['northing'][:])
            self.ymax = np.max(netcdf_dataset['northing'][:])

        else:
            self.data = None

    def sort_variables(self, var_dict):
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

    def grid_sections(self, variables, lines, xres, yres, resampling_method = 'cubic', return_interpolated = False, save_hdf5 = True, hdf5_dir = None):
        """A function for gridding AEM inversoin variables into sections.
           This method can handle both 1D and 2D variables

        Parameters
        ----------
        variables : list of strings
            List of inversions variables from netcdf dataset.
        lines : list of integers
            List of AEM line numbers to grid.
        xres : float
            X-resolution (m) along line.
        yres : float
            Y-resolution (m) along line.
        resampling_method : string
            Method from scipy interpolators. One of
            (‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’,
             ‘previous’, ‘next’)
        return_interpolated : boolean
            If True there will be a class variables for the gridded variables
             for each line
        save_hdf5 : boolean
            If True, we will save the gridded variables as a hdf5 file.
        hdf5_dir : string
            Path to directory in which the hdf5 files are to be saved.


        Returns
        -------
        dictionary
            dictionary with interpolated variables as numpy arrays

        """
        # Check some of the arguments to ensure they are lists

        lines = check_list_arg(lines)
        variables = check_list_arg(variables)


        # Add key variables if they aren't in the list to grid
        for item in ['easting', 'northing', 'elevation', 'fiducial', 'layer_top_depth', 'layer_centre_depth']:
            if np.logical_and(item not in variables, item in self.data.variables):
                variables.append(item)

        self.section_variables = variables

        # First create generators for returning coordinates and variables for the lines

        cond_lines= get_lines(self.data,
                              line_numbers=lines,
                              variables=self.section_variables)

        # Interpolated results will be added to a dictionary
        interpolated = {}

        # Create a gridding parameters dictionary

        gridding_params = {'xres': xres, 'yres': yres,
                           'resampling_method': resampling_method}

        # Iterate through the lines
        for i in range(len(lines)):

            # Extract the variables and coordinates for the line in question
            line_no, cond_var_dict = next(cond_lines)

            # Now we need to sort the cond_var_dict and run it east to west
            cond_var_dict = self.sort_variables(cond_var_dict)

            # If there is no 'layer_top_depth' add it
            if np.logical_and('layer_top_depth' not in cond_var_dict,
                              'layer_centre_depth' in cond_var_dict):

                cond_var_dict['layer_top_depth'] = spatial_functions.layer_centre_to_top(cond_var_dict['layer_centre_depth'])
                #del cond_var_dict['layer_centre_depth']

            interpolated[line_no] =  self.grid_variables(line_no, cond_var_dict,
                                                         gridding_params)
            # Save to hdf5 file if the keyword is passed
            if save_hdf5:
                fname = os.path.join(hdf5_dir, str(int(line_no)) + '.hdf5')
                dict_to_hdf5(fname, interpolated[line_no])

            # Many lines may fill up memory so if the dictionary is not being returned then
            # we garbage collect
            if not return_interpolated:

                del interpolated[line_no]

                # Collect the garbage
                gc.collect()

        if return_interpolated:
            self.section_data = interpolated
        else:
            self.section_data = None

    def grid_variables(self, line, cond_var_dict, gridding_params):
        """Function controlling the vertical gridding of 2D and 1D variables.

        Parameters
        ----------
        line : int
            line number.
        cond_var_dict : dictionary
            dictionary of variables to be gridded.
        gridding_params : dictionary
            parameters for interpolation.

        Returns
        -------
        dictionary
            Dictionary of inteprolated variables
        """

        # Create an empty dictionary
        interpolated = {}

        # Create a sort mask in cases where the lines are not in order

        # Define coordinates
        utm_coordinates = np.column_stack((cond_var_dict['easting'],
                                          cond_var_dict['northing']))

        # Add the flag to the dictionary
        #if utm_coordinates[0, 0] > utm_coordinates[-1, 0]:
        #    cond_var_dict['reverse_line'] = True
        #else:
        #    cond_var_dict['reverse_line'] = False

        # Add distance array to dictionary
        cond_var_dict['distances'] = spatial_functions.coords2distance(utm_coordinates)

        # Add number of epth cells to the array
        if 'depth' in self.data.dimensions:
            cond_var_dict['ndepth_cells'] = self.data.dimensions['depth'].size
        else:
            cond_var_dict['ndepth_cells'] = self.data.dimensions['layer'].size

        # Interpolate 2D and 1D variables

        vars_2d = [v for v in self.section_variables if cond_var_dict[v].ndim == 2]
        vars_1d = [v for v in self.section_variables if cond_var_dict[v].ndim == 1]

        # Generator for inteprolating 2D variables from the vars_2d list
        interp2d = spatial_functions.interpolate_2d_vars(vars_2d, cond_var_dict,
                                                        gridding_params['xres'],
                                                        gridding_params['yres'])
        for var in vars_2d:
            # Generator yields the interpolated variable array
            interpolated[var], cond_var_dict = next(interp2d)


        # Add grid distances and elevations to the interpolated dictionary
        interpolated['grid_distances'] = cond_var_dict['grid_distances']
        interpolated['grid_elevations'] = cond_var_dict['grid_elevations']

        # Generator for inteprolating 1D variables from the vars_1d list
        interp1d = spatial_functions.interpolate_1d_vars(vars_1d, cond_var_dict,
                                       gridding_params['resampling_method'])

        for var in vars_1d:
            # Generator yields the interpolated variable array
            interpolated[var] = next(interp1d)

        return interpolated

    def load_gridded_sections(self, f, gridded_vars):
        """Pull data from h5py object to a dictionary

        Parameters
        ----------
        f : object
            A h5py open file
        plot_vars : sequence
            Sequence of variables to load

        Returns
        -------
        dictionary
            Gridded variables.
        """
        # Create empty dictionary
        datasets = {}
        # Iterate through h5py objects
        for item in f.values():
            if item.name[1:] in gridded_vars:
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
    def load_lci_layer_grids_from_pickle(self, pickle_file):
        """This is a hack to remove the need for rasterio.
        We have preloaded the data into a pickle file

        Parameters
        ----------
        pickle_file : string
            path to pickle file
        Returns
        -------
        self, dictionary
            dictionary with layer grids and metadata

        """

        layer_grids = pickle.load( open( pickle_file, "rb" ) )
        self.layer_grids = layer_grids


    def load_sections_from_file(self, hdf5_dir, grid_vars, lines = []):
        """Load pre-gridded AEM sections from file.

        Parameters
        ----------
        hdf5_dir : string
            Path to hdf5 files.

        grid_vars : list
            A list of variables to load from hdf5 files

        Returns
        -------
        self, dictionary
            Python dictionary with gridded line data

        """
        interpolated = {}
        # iterate through the files
        if lines == []:
            for file in glob.glob(os.path.join(hdf5_dir, '*.hdf5')):
                line = int(os.path.basename(file).split('.')[0])
                lines.append(line)

        for line in lines:

            file = os.path.join(hdf5_dir, str(line) + '.hdf5')

            f = h5py.File(file, 'r')

            interpolated[line] = extract_hdf5_data(f, grid_vars)

            f.close()

        self.section_data = interpolated

    def create_flightline_polylines(self):
        """
        Create polylines from the AEM flight lines
        Returns
        self, dictionary
           dictionary of polyline with line number as the key
        -------

        """
        assert np.logical_and("line" in self.data.variables,
                              "line_index" in self.data.variables)
        self.flight_lines = {}

        for i, line in enumerate(self.data['line'][:]):
            mask = np.where(self.data['line_index'][:] == i)
            # First sort by fiducial
            sort_mask = np.argsort(self.data['fiducial'][mask])
            # now get the easting and northing and sort
            easting = self.data['easting'][mask][sort_mask]
            northing = self.data['northing'][mask][sort_mask]

            # add the polyline to the attribute
            self.flight_lines[line] = LineString(np.column_stack((easting, northing)))

class AEM_data:
    """
    Class for handling AEM inversions
    """

    def __init__(self, name = '', system_name = '', netcdf_dataset = None):
        """Initialise instance of AEM data class.
        Parameters
        ----------
        inversion_type : string
            One of 'deterministic' or 'stochastic'
        netcdf_dataset : type
            netcdf dataset contain AEM data
        Returns
        -------
        type
            Description of returned object.

        """
        self.name = name
        self.system_name = system_name

        # Check netcdf file
        if netcdf_dataset is not None:
            if testNetCDFDataset(netcdf_dataset):
                self.data = netcdf_dataset
            else:
                raise ValueError("Input datafile is not netCDF4 format")
                return None
            # Create an instance variable of the coordinates
            self.coords = np.column_stack((netcdf_dataset['easting'][:],
                                           netcdf_dataset['northing'][:])).data
            # Get some of the usefule metadata from the netcdf file
            self.xmin = np.min(netcdf_dataset['easting'][:])
            self.xmax = np.max(netcdf_dataset['easting'][:])
            self.ymin = np.min(netcdf_dataset['northing'][:])
            self.ymax = np.max(netcdf_dataset['northing'][:])

        else:
            self.data = None

    def calculate_additive_noise(self, aem_gate_data, high_altitude_mask):
        """Function for calculating the additive noise from high altitude lines.

        Parameters
        ----------
        aem_gate_data: array
            array with AEM gate data
        high_altitude_mask : boolean array

        Returns
        -------
        array
            Numpy array with an estimate of additive noise for each gate

        """
        # In case of negative
        high_alt_data = aem_gate_data[high_altitude_mask,:]
        arr = np.sqrt(np.std(high_alt_data, axis = 0)**2)

        # Now repeat to size of gate dat array
        return np.repeat(arr[np.newaxis,:], aem_gate_data.shape[0], axis=0)



    def calculate_noise(self, data_variable, noise_variable = None, multiplicative_noise = 0.03, high_altitude_lines = None):
        """A function for calculating the noise for AEM data.

        Parameters
        ----------
        data_variable : string
            NetCDF variable name for the EM data. If the AEM system is a dual
            moment system, then this should be just one of the moments.
        noise_variable : string
            The attribute name for your noise. If this is none then the noise
            will be named the data variable + '_noise'
        multiplicative_noise : float
            Fraction defining the additive_noise. By default we use 3% or 0.03
            * the AEM data but this will vary from system to system
        high_altitude_lines : array
            An array with  high altitude line numbers
            If none we will assume high altitude lines start with 913

        Returns
        -------
        self, array
            An array of EM noise estimates

        """
        if noise_variable is None:
            noise_variable = data_variable + "_noise"
        if high_altitude_lines is None:
            high_altitude_lines =  [x for x in self.data['line'][:] if x>913000]
        # Get a high alitute line mask

        high_altitude_mask = get_lookup_mask(high_altitude_lines, self.data)

        # Get the AEM data
        aem_gate_data = self.data[data_variable][:].data


        # Calculate the additive noies
        additive_noise_arr = self.calculate_additive_noise(aem_gate_data, high_altitude_mask)

        # Calculate multiplicative noise
        mulitplicative_noise_arr = multiplicative_noise * aem_gate_data

        # Get sum of squares of two noise sources
        noise =  np.sqrt(mulitplicative_noise_arr**2 + additive_noise_arr**2)

        setattr(self, noise_variable, noise)
