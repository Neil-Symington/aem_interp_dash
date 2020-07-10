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

import netCDF4
import numpy as np
import pandas as pd
import spatial_functions
import netCDF4
from netcdf_utils import get_lines, testNetCDFDataset
from misc_utils import check_list_arg, dict_to_hdf5
import gc, glob, os
import rasterio
import tempfile
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from shapely.geometry import Point, MultiPoint, Polygon

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
                    self.conductivity_model = netcdf_dataset
                else:
                    raise ValueError("Input datafile is not netCDF4 format")
                    return None

            else:
                self.conductivity_model = None
            # Create an instance variable of the coordinates
            self.coords = np.column_stack((netcdf_dataset['easting'][:],
                                           netcdf_dataset['northing'][:]))

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
            for item in ['easting', 'northing', 'elevation', 'layer_top_depth']:
                if item not in variables:
                    variables.append(item)

            # First create generators for returning coordinates and variables for the lines

            cond_lines= get_lines(self.conductivity_model,
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

                cond_var_dict['utm_coordinates'] = np.column_stack((cond_var_dict['easting'],
                                                                    cond_var_dict['northing']))

                interpolated[line_no] =  self.grid_conductivity_variables(line_no, cond_var_dict,
                                                                          gridding_params)

                # Save to hdf5 file if the keyword is passed
                if save_hdf5:
                    fname = os.path.join(hdf5_dir, str(line_no) + '.hdf5')
                    if overwrite_hdf5:
                        dict_to_hdf5(fname, interpolated[line_no])
                    else:
                        if os.path.exists(fname):
                            print("File ", fname, " already exists")
                        else:
                            dict_to_hdf5(fname, interpolated[line_no])

                # Many lines may fill up memory so if the dictionary is not being returned then
                # we garbage collect
                if not return_dict:

                    del interpolated[line_no]

                    # Collect the garbage
                    gc.collect()

            if return_interpolated:
                self.section_variables = interpolated
            else:
                self.section_variables = None

        def grid_variables(self, line, cond_var_dict, gridding_params):
            """Function controlling the vertical gridding of 2D and 1D variable
            s.

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
                Ddictionary of inteprolated variables
            """
            # Create path if it doesn't exist
            if not os.path.exists(hdf5_dir):
                os.mkdir(hdf5_dir)

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
            cond_var_dict['distances'] = spatial_functions.coords2distance(utm_coordinates)

            # Add number of layers to the array
            cond_var_dict['nlayers'] = self.conductivity_model.dimensions['layer'].size

            # Interpolate 2D and 1D variables

            vars_2d = [v for v in self.conductivity_variables if cond_var_dict[v].ndim == 2]
            vars_1d = [v for v in self.conductivity_variables if cond_var_dict[v].ndim == 1]

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

        def load_lci_layer_grids(self, inDir, conversion_to_SI = True, nlayers = 30):
            """A function for loading layer lci layer grids. This will only
            work if the file names follow this particular convention
            ##TODO make a more general function

            Parameters
            ----------
            inDir : string
                path to directory with grids
            conversion_to_SI : boolean
                Description of parameter `conversion_to_SI`.

            Returns
            -------
            self, dictionary
                dictionary with layer grids and metadata

            """
            assert os.path.exists(inDir)
            inDir = os.path.join(inDir, "*.ers")

            # Dictionary to write results into
            layer_grids = {}

            for file in glob.glob(inDir):
                layer = int(file.split('Con')[1].split('_')[0])
                if not layer == nlayers:
                    depth_from = float(file.split("gm_")[-1].split('.ers')[0].split('-')[0])
                    depth_to = float(file.split("gm_")[-1].split('.ers')[0].split('-')[1].split('m')[0])
                else:
                    depth_from = float(file.split("gm_")[-1].split('.ers')[0].split('-')[0].split('m+')[0])

                cond_dataset = rasterio.open(file)
                arr = cond_dataset.read(1)
                arr[arr == cond_dataset.get_nodatavals()] = np.nan
                # convert to S/m
                if conversion_to_SI:
                    arr = arr/1000.
                key = "Layer_" + str(layer)
                layer_grids[key] = {}
                layer_grids[key]['conductivity'] = arr
                layer_grids[key]['depth_from'] = depth_from
                layer_grids[key]['depth_to'] = depth_to

            layer_grids['raster_transform'] = cond_dataset.transform
            # make bounds similar to extent in matplotlib for ease of plotting
            bounds = [cond_dataset.bounds.left, cond_dataset.bounds.right,
                      cond_dataset.bounds.bottom, cond_dataset.bounds.top]

            layer_grids['bounds'] = bounds

            self.layer_grids = layer_grids

class modelled_boundary:
    """
    Class for handling interpreted stratigraphic boundaries
    """
    def __init__(self, name = None, outfile_path = None):
        """Initialise instance of modelled boundary.

        Parameters
        ----------
        name : type
            Description of parameter `name`.
        outfile_path: string
            Path for output csv file

        """
        if name is not None:
            self.name = name
        else:
            self.name = "Unnamed_boundary"
        # Create points dataset
        self.interpreted_points = {'fiducial': [],
                                   'easting': [],
                                   'northing': [],
                                   'layer_depth': [],
                                   'layer_elevation': [],
                                   'standard_deviation': []}
        if outfile_path is not None:
            assert os.path.exists(os.path.dirname(outfile_path))
            assert outfile_path.split('.')[-1] == 'csv'
            self.outfile_path = outfile_path
        else:
            outdir = tempfile.gettempdir()
            self.outfile_path = os.path.join(outdir,
                                             self.name + "_interpreted_points.csv")
        # Create an instance variable of grid coordinates
        self.grid_coords = None
    def get_convex_hull(self, convex_hull_buffer = 1000.):
        """
        Function for finding the convex hull of our boundary given current
        point interpretations.

        convex_hull_buffer: float
            The buffer around the convex hull for masking our interpolated surface

        Returns:

        shapely polygon
            convex hull
        """
        coords = np.column_stack((self.interpreted_points['easting'],
                                 self.interpreted_points['northing']))

        points = [Point(pt) for pt in coords]
        mpt = MultiPoint(points)
        return mpt.convex_hull.buffer(convex_hull_buffer)

    def save_points(self):
        """
        Function for saving the current interpreted points to
        """
        pd.DataFrame(interpretations, index = 'fiducial').transpose().to_csv(self.outfile_path)

    def load_interpretation_points_from_file(self, infile_path):
        """Function for loading previously saved interpretation points.

        Parameters
        ----------
        infile_path : string
            path to saved points.

        Returns
        -------
        self, dictionary
            dictionary with interpretations

        """
        df = pd.read_csv(infile_path)

        for key in self.interpreted_points.keys():
            self.interpreted_points[key] += df[key].tolist()


    def create_interpolator(self, length_scale):
        """Create an Gaussian interpolator for on the fly gridding.

        ##TOD implement way to have more than one interpolator

        Parameters
        ----------
        length_scale : float
            Length scale for interpolation.  This is very dependent on the data
            desnity and the smoothness of the boudnary

        Returns
        -------
        object
            Gaussian process instance.

        """
        kernel = Matern(length_scale=length_scale)
        self.gaussian_process = GaussianProcessRegressor(kernel=kernel,
                                                         n_restarts_optimizer=10)
    def create_grid(self, xmin, xmax, ymin, ymax, cell_size = 500.):
        """Function for defining a grid for interpolating the boundary onto

        Parameters
        ----------
        xmin : float
            Description of parameter `xmin`.
        xmax : float
            Description of parameter `xmax`.
        ymin : float
            Description of parameter `ymin`.
        ymax : float
            Description of parameter `ymax`.
        cell_size : float
            Description of parameter `cell_size`.
        Returns
        -------
        grid_coords
            2d array of grid coordinates.
        """
        # Create grid of x and y coordinates
        self.bounds = [xmin, xmax, ymin, ymax]
        self.cell_size = cell_size
        x__, y__ = np.mgrid[xmin:xmax:cell_size, ymax:ymin:-1*cell_size]
        self.width = x__.shape[1]
        self.height = x__.shape[0]
        self.grid_coords = np.column_stack((x__.ravel(), y__.ravel()))

    def fit_interpolator(self, variable, length_scale = 500.):
        """Fit the gaussian process to generate a function for prediction.

        Parameters
        ----------
        variable : string
            Variable from the self.interpreted_points.
        length_scale : float
            Length scale for interpolation.  This is very dependent on the data
            desnity and the smoothness of the boudnary
        """
        assert variable in self.interpreted_points.keys()
        self.interpolator_variable = variable
        # Create instance on Gaussian process
        self.create_interpolator(length_scale=length_scale)

        # Fit the interpolator

        X = np.column_stack((self.interpreted_points['easting'],
                            self.interpreted_points['northing']))

        y = self.interpreted_points[variable]

        self.gaussian_process.fit(X,y)
    def predict_at_points(self, coordinates):
        """A function for predicting the from our Gaussian process.

        Parameters
        ----------
        coordinates : ndarray of shape (n_samples, 2)

        Returns
        -------
        y_mean
            An array of predictions at the input coordinates

        """
        return self.gaussian_process.predict(coordinates)
    def predict_on_grid(self, convex_hull_buffer = 1000.):
        """A function for predicting onto every point on our grid.

        convex_hull_buffer: float
            The buffer around the convex hull for masking our interpolated surface

        Returns
        -------
        ndarray
            2D gridded array

        """
        if self.grid_coords is None:
            raise ValueError("Define grid coordinates")
            return None
        else:
            grid = self.predict_at_points(self.grid_coords)

            # create a mask showing which points are within the
            # polygon
            poly = self.get_convex_hull(convex_hull_buffer = 1000.)
            mask = np.array([Point(x).within(poly) for x in self.grid_coords])

            # Assign all values outside of the convex hull nan
            grid = grid.flatten()
            grid[~mask] = np.nan
            # reshape for easy plotting
            self.interpreted_grid = grid.reshape((self.height,self.width))#.T
