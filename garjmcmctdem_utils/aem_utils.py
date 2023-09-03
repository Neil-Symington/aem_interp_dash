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
import numpy as np
from garjmcmctdem_utils import spatial_functions, misc_utils
from garjmcmctdem_utils.netcdf_utils import get_lines, testNetCDFDataset, get_lookup_mask
import gc, glob, os
from shapely.geometry import LineString
import geopandas as gpd
import re
import xarray

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
    def getVarsByLine(self, lineNumber=None, variables=None):
        lineInd = np.where(self.data['line'][:] == lineNumber)[0]
        lineMask = np.where(self.data['line_index'][:] == lineInd)[0]
        varDict = {}
        for var in variables:
            varDict[var] = self.data[var][lineMask]
        return varDict


    def grid_sections(self, geometry_variables, inversion_variables, data_variables,
                      lines, xres, yres, resampling_method = 'linear', return_interpolated = False,
                      save_to_disk = True, output_dir = None, sort_on = 'easting'):
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
        save_to_disk : boolean
            If True, we will save the gridded variables as a pickle file.
        output_dir : string
            Path to directory in which the hdf5 files are to be saved.


        Returns
        -------
        dictionary
            dictionary with interpolated variables as numpy arrays

        """
        # Check some of the arguments to ensure they are lists

        lines = misc_utils.check_list_arg(lines)
        self.geometry_variables = misc_utils.check_list_arg(geometry_variables)
        self.inversion_variables = misc_utils.check_list_arg(inversion_variables)
        self.data_variables = misc_utils.check_list_arg(data_variables)

        # Add key variables if they aren't in the list to grid
        for item in ['easting', 'northing', 'elevation', 'fiducial', 'layer_top_depth', 'layer_centre_depth']:
            if np.logical_and(item not in geometry_variables, item in self.data.variables):
                geometry_variables.append(item)

        self.section_variables = self.geometry_variables + self.inversion_variables + self.data_variables

        # Interpolated results will be added to a dictionary
        interpolated = {}

        # Create a gridding parameters dictionary

        gridding_params = {'xres': xres, 'yres': yres,
                           'resampling_method': resampling_method}

        # Iterate through the lines
        for i, line_no in enumerate(lines):

            # Extract the variables and coordinates for the line in question
            cond_var_dict = self.getVarsByLine(lineNumber=line_no, variables=self.section_variables)

            # Now we need to sort the cond_var_dict and run it in a specific direction
            cond_var_dict = spatial_functions.sort_variables(cond_var_dict, sort_on = sort_on)

            # If there is no 'layer_top_depth' add it
            if np.logical_and('layer_top_depth' not in cond_var_dict,
                              'layer_centre_depth' in cond_var_dict):

                cond_var_dict['layer_top_depth'] = spatial_functions.layer_centre_to_top(cond_var_dict['layer_centre_depth'])

            interpolated[line_no] =  self.grid_variables(cond_var_dict, gridding_params)

            # Save to hdf5 file if the keyword is passed
            if save_to_disk:
                fname = os.path.join(output_dir, str(int(line_no)) + '.pkl')
                file = open(fname, 'wb')
                # dump information to the file
                pickle.dump(interpolated[line_no], file)

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

    def grid_variables(self, cond_var_dict, gridding_params):
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

        # Add distance array to dictionary
        cond_var_dict['distances'] = spatial_functions.coords2distance(utm_coordinates)

        # Add number of depth cells to the array
        if 'depth' in self.data.dimensions:
            cond_var_dict['ndepth_cells'] = self.data.dimensions['depth'].size
        else:
            cond_var_dict['ndepth_cells'] = self.data.dimensions['layer'].size

        # Interpolate 2D and 1D variables
        vars_2d = [v for v in self.inversion_variables if cond_var_dict[v].ndim == 2]
        vars_1d = [v for v in self.section_variables if cond_var_dict[v].ndim == 1]
        vars_1p5d = [v for v in self.data_variables if cond_var_dict[v].ndim == 2]

        # Generator for interpolating 2D variables from the vars_2d list
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

        # Now we griddify the data variables
        interpolated_utm = np.column_stack((interpolated['easting'], interpolated['northing']))

        interp1p5d = spatial_functions.interpolate_data(vars_1p5d, cond_var_dict, interpolated_utm)

        for var in vars_1p5d:
            interpolated[var] = next(interp1p5d)

        # Create an xarray from the dictionary

        coords = {}
        data_vars = {}

        coords['grid_distances'] = interpolated['grid_distances']
        coords['grid_elevations'] = interpolated['grid_elevations']
        if len(self.data_variables) > 0:
            coords['windows'] = np.arange(1,interpolated[self.data_variables[0]].shape[1]+1)

        var_list = [e for e in interpolated.keys() if e not in coords.keys()]

        for var in var_list:

            if len(interpolated[var].shape) == 2:
                if var in self.inversion_variables:
                    data_vars[var] = (['grid_elevations', 'grid_distances'], interpolated[var])
                elif var in self.data_variables:
                    data_vars[var] = (['grid_distances', 'windows'], interpolated[var])

            elif len(interpolated[var].shape) == 1:
                data_vars[var] = (['grid_distances'], interpolated[var])

        ds = xarray.Dataset(data_vars, coords=coords)

        return ds

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

    def create_flightline_polylines(self, crs):
        """
        Create polylines from the AEM flight lines
        Returns
        self, dictionary
           dictionary of polyline with line number as the key
        -------

        """
        assert np.logical_and("line" in self.data.variables,
                              "line_index" in self.data.variables)
        flight_lines = {}

        for i, line in enumerate(self.data['line'][:]):
            mask = np.where(self.data['line_index'][:] == i)
            # First sort by fiducial
            sort_mask = np.argsort(self.data['fiducial'][mask])
            # now get the easting and northing and sort
            easting = self.data['easting'][mask][sort_mask]
            northing = self.data['northing'][mask][sort_mask]

            # add the polyline to the attribute
            flight_lines[line] = LineString(np.column_stack((easting, northing)))

        self.flightlines = gpd.GeoDataFrame(data = {'lineNumber': flight_lines.keys(),
                                             'geometry': flight_lines.values()},
                                            geometry= 'geometry', crs = crs)

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
        self.section_variables = None

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

    def calculate_noise(self, data_variable, noise_variable = None, additive_noise_variable = None,
                        multiplicative_noise = 0.03, high_altitude_lines = None):
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
            noise_variable = data_variable.replace('-', '_') + "_noise"
        if additive_noise_variable is None:
            additive_noise_variable = data_variable.replace('-', '_') + "_additive_noise"
        if high_altitude_lines is None:
            high_altitude_lines =  [x for x in self.data['line'][:] if x>913000]
        # Get a high alitute line mask

        high_altitude_mask = get_lookup_mask(high_altitude_lines, self.data)

        # Get the AEM data
        aem_gate_data = self.data[data_variable][:].data


        # Calculate the additive noies
        additive_noise_arr = self.calculate_additive_noise(aem_gate_data, high_altitude_mask)

        # Set the additive noise as an attribute
        setattr(self, additive_noise_variable, (additive_noise_arr[0]))

        # Calculate multiplicative noise
        mulitplicative_noise_arr = multiplicative_noise * aem_gate_data

        # Get sum of squares of two noise sources
        noise =  np.sqrt(mulitplicative_noise_arr**2 + additive_noise_arr**2)

        setattr(self, noise_variable, noise)

    def griddify_variables(self, variables, lines,  return_gridded = False, save_to_disk = True, output_dir = None):
        # Check some of the arguments to ensure they are lists

        lines = misc_utils.check_list_arg(lines)
        variables = misc_utils.check_list_arg(variables)

        # Add key variables if they aren't in the list to grid
        for item in ['easting', 'northing', 'elevation', 'fiducial']:
           if np.logical_and(item not in variables, item in self.data.variables):
               variables.append(item)

        self.section_variables = variables # consider removing

        # First create generators for returning coordinates and variables for the lines

        em_lines = get_lines(self.data,
                              line_numbers=lines,
                              variables=self.section_variables)

        # Interpolated results will be added to a dictionary
        griddified = {}

        # Create a gridding parameters dictionaryd}

        # Iterate through the lines
        for i in range(len(lines)):

            # Extract the variables and coordinates for the line in question
            line_no, em_var_dict = next(em_lines)

            # Now we need to sort the cond_var_dict and run it east to west
            em_var_dict = spatial_functions.sort_variables(em_var_dict)

            # add grid distances so we can plot it in a line

            utm_coords = np.column_stack((em_var_dict['easting'], em_var_dict['northing']))

            em_var_dict['grid_distances'] = spatial_functions.coords2distance(utm_coords)

            griddified[line_no] = misc_utils.dict2xr(em_var_dict, dims=['grid_distances'])

            if save_to_disk:
                fname = os.path.join(output_dir, str(int(line_no)) + '.pkl')
                file = open(fname, 'wb')
                # dump information to the file
                pickle.dump(griddified[line_no], file)

            # Many lines may fill up memory so if the dictionary is not being returned then
            # we garbage collect
            if not return_gridded:
                del griddified[line_no]

                # Collect the garbage
                gc.collect()

        if return_gridded:
            self.section_data = griddified
        else:
            self.section_data = None


# Class for extracting regular expressions from the stm files
class _RegExLib:
    """Set up regular expressions"""
    # use https://regexper.com to visualise these if required
    _reg_begin = re.compile(r'(.*) Begin\n')
    _reg_end = re.compile(r'(.*) End\n')
    _reg_param = re.compile(r'(.*) = (.*)\n')

    __slots__ = ['begin', 'end', 'param']

    def __init__(self, line):
        # check whether line has a positive match with all of the regular expressions
        self.begin = self._reg_begin.match(line)
        self.end = self._reg_end.match(line)
        self.param = self._reg_param.match(line)



# Define the blocks from the stm files

blocks = {'Transmitter': ['NumberOfTurns', 'PeakCurrent', 'LoopArea',
                          'BaseFrequency', 'WaveformDigitisingFrequency',
                          'WaveFormCurrent'],
          'Receiver': ['NumberOfWindows', 'WindowWeightingScheme',
                       'WindowTimes', 'CutOffFrequency', 'Order'],

          'ForwardModelling': ['ModellingLoopRadius', 'OutputType',
                               'SaveDiagnosticFiles', 'XOutputScaling',
                               'YOutputScaling', 'ZOutputScaling',
                               'SecondaryFieldNormalisation',
                               'FrequenciesPerDecade',
                               'NumberOfAbsiccaInHankelTransformEvaluation']}

class AEM_System:

    def __init__(self, name, dual_moment=True):
        """
        :param name: string: system name
        :param dual_moment: boolean, is the system fual moment (i.e. syktem
        """

        self.name = name

        if dual_moment:
            self.LM = {'Transmitter': {}, 'Receiver': {}, 'ForwardModelling': {}}
            self.HM = {'Transmitter': {}, 'Receiver': {}, 'ForwardModelling': {}}

    def parse_stm_file(self, infile, moment):

        # Save the results into a dictionary

        parameters = {}
        # Extract file line by line
        with open(infile, 'r') as f:
            # Yield the lines from the file
            line = next(f)
            while line:
                reg_match = _RegExLib(line)

                if reg_match.begin:
                    key = reg_match.begin.group(1).strip()

                    if key == "WaveFormCurrent":
                        a = misc_utils.block_to_array(f)
                        parameters[key] = a

                    if key == "WindowTimes":
                        a = misc_utils.block_to_array(f)
                        parameters[key] = a

                if reg_match.param:
                    key = reg_match.param.group(1).strip()
                    val = reg_match.param.group(2).strip()

                    if misc_utils.RepresentsInt(val):
                        val = int(val)

                    elif misc_utils.RepresentsFloat(val):
                        val = float(val)

                    elif key == "CutOffFrequency":

                        val = np.array([int(x) for x in val.split()])

                    if not key.startswith(r'//'):
                        parameters[key] = val

                line = next(f, None)

        for item in blocks.keys():
            for entry in blocks[item]:
                if moment == "HM":
                    self.HM[item][entry] = parameters[entry]
                elif moment == "LM":
                    self.LM[item][entry] = parameters[entry]