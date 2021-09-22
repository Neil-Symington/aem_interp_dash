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
Created on 16/7/2019
@author: Neil Symington

Code for creating geological modelled objects

'''
import os
import tempfile
import numpy as np
from numpy.core.fromnumeric import var
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from shapely.geometry import Point, MultiPoint, Polygon
import geopandas as gpd

class Statigraphic_Model:
    """
    Class for handling a full stratigraphic model
    """
    def __init__(self, name = None, outfile_path = None, existing_interpretation_file = None):
        if name is not None:
            self.name = name
        else:
            self.name = "Unnamed_model"
        # Create points dataset
        if existing_interpretation_file is not None:
            self.interpreted_points = pd.read_csv(existing_interpretation_file)

        if outfile_path is not None:
            assert os.path.exists(os.path.dirname(outfile_path))
            assert outfile_path.split('.')[-1] == 'csv'
            self.outfile_path = outfile_path
        else:
            outdir = tempfile.gettempdir()
            self.outfile_path = os.path.join(outdir, self.name + "_interpreted_points.csv")
        # create attribute with surface names
        self.surfaces = []
    def initiatialise_surfaces_from_template(self, df_template):
        """
        A function for initialising models using a template csv file

        Parameters
        ----------
        infile     dataframe
        valid path to csv file with important model information

        Returns
        -------

        """
        # define mandatory columns
        mandatory_columns =['SurfaceName','Type','BoundaryNm']

        assert np.isin(mandatory_columns, df_template.columns).all()
        assert len(df_template['SurfaceName'].unique()) == len(df_template)
        ## raise an exception for asserts

        # iterate through the rows and create instances of the model class
        for index, row in df_template.iterrows():
            self.initiatialise_surface(row)

    def initiatialise_surface(self, series):
        """
        """
        surfaceName = series['SurfaceName']
        # initiate model using model boundary class
        surface = model_boundary(name=surfaceName)
        setattr(self, surfaceName, surface)
        if not surfaceName in self.surfaces:
            self.surfaces.append(surfaceName)
        # iterate through axes in data series and set them as attributes
        for col in series.axes[0]:
            if not col == 'SurfaceName':
                setattr(getattr(self, surfaceName), col, series[col])
        # add interpreted points to the model boundary
        if hasattr(self, "interpreted_points"):
            setattr(getattr(self, surfaceName), "interpreted_points",
                     self.interpreted_points[self.interpreted_points['SurfaceName'] == surfaceName])



class model_boundary:
    """
    Class for handling interpreted stratigraphic boundaries
    """
    def __init__(self, name = None):
        """Initialise instance of modelled boundary.

        Parameters
        --------

        """
        if name is not None:
            self.name = name
        else:
            self.name = "Unnamed_boundary"

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
        coords = np.column_stack((self.interpreted_points['X'],
                                 self.interpreted_points['Y']))

        points = [Point(pt) for pt in coords]
        mpt = MultiPoint(points)
        return mpt.convex_hull.buffer(convex_hull_buffer)

    def create_interpolator(self, kernel = Matern(length_scale=5000, nu = 1.5), name = 'interpolator_1'):
        """Create an Gaussian interpolator for on the fly gridding.

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
        # first check the interpolator name is original
        if hasattr(self, name):
            print('That interpolator name is in use. Please use another.')
            raise ValueError()

        setattr(self, name, GaussianProcessRegressor(kernel=kernel,
                                                     n_restarts_optimizer=5,
                                                     normalize_y=True))

    def create_grid(self, xmin, xmax, ymin, ymax, cell_size = 500., convex_hull = False,convex_hull_buffer = 1000.):

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

        # Create a convex hull
        if convex_hull:
            self.convex_hull = self.get_convex_hull(convex_hull_buffer = convex_hull_buffer)

    def fit_interpolator(self, interpolator_name, variable = None, arr = None):
        """Fit the gaussian process to generate a function for prediction.

        Parameters
        ----------
        """
        # first check the interpolator exists
        if not hasattr(self, interpolator_name):
            print('That interpolator does not exist. Please create one or check the interpolator_name keyword argument.')
            raise ValueError()
        
        if variable is not None:
            assert variable in self.interpreted_points.keys()
            y = self.interpreted_points[variable]
        elif arr is not None:
            y = arr
        else:
            return None
        

        # Fit the interpolator

        X = np.column_stack((self.interpreted_points['X'],
                             self.interpreted_points['Y']))


        gp = getattr(self,interpolator_name)
        gp.fit(X,y)

    def predict_at_points(self, coordinates, interpolator_name, return_std = True):
        """A function for predicting the from our Gaussian process.

        Parameters
        ----------
        coordinates : ndarray of shape (n_samples, 2)

        Returns
        -------
        y_mean
            An array of predictions at the input coordinates

        """
        gp = getattr(self,interpolator_name)
        return gp.predict(coordinates, return_std = return_std)

    def predict_on_grid(self, interpolator_name, grid_name, return_std = True):
        """A function for predicting onto every point on our grid.

        convex_hull_buffer: float
            The buffer around the convex hull for masking our interpolated surface

        Returns
        -------
        ndarray
            2D gridded array

        """
        spatial_mask = False
        # first check the interpolator exists
        if not hasattr(self, interpolator_name):
            print('That interpolator does not exist. Please create one or check the interpolator_name keyword argument.')
            raise ValueError()

        # Check grid coordinates are defined
        if not self.grid_coords is None:
            if return_std:
                grid, grid_std = self.predict_at_points(self.grid_coords, interpolator_name)
            else:
                grid = self.predict_at_points(self.grid_coords, interpolator_name, return_std = False)

            # If a convex hull exists then we will use it to create a mask for
            # our inteprolated grid
            if hasattr(self, 'extent'):
                poly = self.extent
                spatial_mask = True
            elif hasattr(self, 'convex_hull'):
                poly = self.convex_hull
                spatial_mask = True
            if spatial_mask:
                mask = np.array([Point(x).within(poly) for x in self.grid_coords])
                # Assign all values outside of the convex hull nan
                grid = grid.flatten()
                grid[~mask] = np.nan
                # reshape for easy plotting
                grid = grid.reshape((self.height,self.width)).T

                ## TODO add flag
                if return_std:
                    grid_std = grid_std.flatten()
                    grid_std[~mask] = np.nan
                    # reshape for easy plotting
            else:
                pass
            setattr(self, grid_name, grid.reshape((self.height,self.width)).T)
            if return_std:
                setattr(self, grid_name + '_std', grid_std.reshape((self.height,self.width)).T)
        else:
            raise ValueError("Define grid coordinates")

    def load_extent_from_file(self, infile, index = 0):
        """A function for loading the extent geometry from a shapefile.
        """
        self.extent = gpd.read_file(infile)['geometry'].values[index]


def full_width_half_max(D, max_idx, fmax):
    """Find the width of the probability interval that is >0.5 times the local
    max probability.

    Parameters
    ----------
    D: dictionary
        Dictionary of rj sounding data
    max_idx : interger
        The inte
    fmax : type
        Description of parameter `fmax`.

    Returns
    -------
    type
        Description of returned object.

    """

    idx_upper = None
    idx_lower = None

    # positive direction
    for idx in np.arange(max_idx, D['depth_cells'].shape[0]):
        if D['change_point_pdf'][idx] <= fmax/2.:
            idx_upper = idx
            break
    # negative direction
    for idx in np.arange(max_idx, -1, -1):
        if D['change_point_pdf'][idx] <= fmax/2.:
            idx_lower = idx
            break
    # Now calculate the width
    if np.logical_and(idx_upper is not None, idx_lower is not None):
        return D['depth_cells'][idx_upper] - D['depth_cells'][idx_lower]
    else:
        return None


def click2estimate(D, yclick, snap_window = 16, stdev_ceiling = 50.):
    """Function for snapping to a layer point probability maximum from a click

    Parameters
    ----------
    D: dictionary
        Dictionary of rj sounding data
    yclick : type
        Description of parameter `yclick`.

    Returns
    -------
    type
        Description of returned object.

    """

    ymin = yclick - snap_window/2
    ymax = yclick + snap_window/2

    # Get the change point probability array for the snap window interval

    idx = np.where(np.logical_and(D['depth_cells']>ymin, D['depth_cells']<ymax))

    # Now find the maximum cpp from this range
    idx_max = np.argmax(D['change_point_pdf'][idx]) + np.min(idx)
    fmax = D['change_point_pdf'][idx_max]
    interpreted_depth = D['depth_cells'][idx_max]

    # from https://en.wikipedia.org/wiki/Full_width_at_half_maximum
    fwhm = full_width_half_max(D, idx_max, fmax)

    if fwhm is not None:
        stdev = fwhm/(2*np.sqrt(2*np.log(2)))
    else:
        stdev = stdev_ceiling
    return interpreted_depth, stdev
