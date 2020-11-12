# Import python modules
import numpy as np
import geopandas as gpd
import netCDF4
import sys, os
import pickle
import xarray as xr
sys.path.append("../scripts")

import spatial_functions
import aem_utils
import netcdf_utils
import modelling_utils
import plotting_functions as plots
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
# Dash dependencies
import plotly.express as px
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64, io


modelName = "Injune"

root ="/home/nsymington/Documents/GA/dash_data"
# path to determinist netcdf file
det_nc_path = os.path.join(root, "Injune_lci_MGA55.nc")
# path to dertiminist grid
det_grid_path = os.path.join(root, "Injune_layer_grids.p")

grid_data = True #If the lci conductivity sections have not yet been gridded then make this flag true
# path to rjmcmcmtdem pmap file
rj_nc_path = os.path.join(root, "Injune_rjmcmc_pmaps.nc")

project_crs = 'EPSG:28353'

lines = [200101, 200401, 200501, 200601, #200701,
         200801,
         200901, 201001, 201101, 201201, 201301, 201401, 201501,
         201601, 201701, 201801, 201901, 202001, 202101, 202201,
         202301, 202401, 202501, 202601, 202701, 202801, 912011]

# Create an instance
lci = aem_utils.AEM_inversion(name = 'Laterally Contrained Inversion (LCI)',
                              inversion_type = 'deterministic',
                              netcdf_dataset = netCDF4.Dataset(det_nc_path))

# Run function
lci.load_lci_layer_grids_from_pickle(det_grid_path)

# Create instance
rj = aem_utils.AEM_inversion(name = 'GARJMCMCTDEM',
                             inversion_type = 'stochastic',
                             netcdf_dataset = netCDF4.Dataset(rj_nc_path))


# Now we have the lines we can grid the lci conductivity data onto vertical grids (known as sections)
# this is the easiest way to visualise the AEM conuctivity in 2-dimensions

# Assign the lci variables to grid
grid_vars = ['conductivity', 'data_residual', 'depth_of_investigation']


# Define the resolution of the sections
xres, yres = 40., 5.

# We will use the lines from the rj



# Define the output directory if saving the grids as hdf plots

out_dir = os.path.join(root, "section_data_lci")

# if the directory doesn't exist, then create it
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if grid_data:
    lci.grid_sections(variables = grid_vars, lines = lines, xres = xres, yres = yres,
                      return_interpolated = False, save_to_disk = True, output_dir = out_dir)

# Grid the rj sections

# Assign the lci variables to grid
grid_vars = ['conductivity_p10', 'conductivity_p50', 'conductivity_p90', 'interface_depth_histogram',
             'misfit_lowest', 'misfit_average']

xres, yres = 80, 2.

# Define the output directory if saving the grids as hdf plots

out_dir = os.path.join(root, "section_data_rj")

# if the directory doesn't exist, then create it
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if grid_data:
    rj.grid_sections(variables = grid_vars, lines = lines, xres = xres, yres = yres,
                      return_interpolated = False, save_to_disk = True, output_dir = out_dir)

# As we want to be able to tie the rj to the lci we need to scale grid distances
def scale_distance_along_line(xarr1, xarr2):
    """

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
    d, i = spatial_functions.nearest_neighbours(new_coords, coords, points_required = 2,max_distance = 250.)

    weights = 1/d

    # Normalise the array so the rows sum to unit

    weights /= np.sum(weights, axis=1)[:, None]

    # get our grid distances
    return np.sum(xarr1['grid_distances'].values[i] * weights, axis=1)



for lin in lines:
    print(lin)
    lci_infile = '/home/nsymington/Documents/GA/dash_data/section_data_lci/'+ str(lin) + '.pkl'

    with open(lci_infile, 'rb') as file:
        xarr1 = pickle.load(file)

    rj_infile = '/home/nsymington/Documents/GA/dash_data/section_data_rj/' + str(lin) + '.pkl'

    with open(rj_infile, 'rb') as file:
        xarr2 = pickle.load(file)

    xarr2['grid_distances'] = scale_distance_along_line(xarr1, xarr2)
    # Save xarray back to pickle file

    file = open(rj_infile, 'wb')
    # dump information to the file
    pickle.dump(xarr2, file)
