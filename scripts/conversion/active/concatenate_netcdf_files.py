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

"""
A script for compiling individual conductivity probability maps netcdf files
that are output by the garjmcmctdem inversion code into a single netcdf file.

## N.B this currently cannot deal with compiling files with varying array lengths.
For example if the user changed the number of layers, samples etc

##TODO implements this
"""


import netCDF4
import os
import numpy as np
import glob
import yaml
import datetime
from pyproj import CRS,Transformer

# Find the file paths from the pmap directory
indir = r"C:\Users\PCUser\Desktop\AEM\rjmcmc\nc_subset"
yml_file = r"C:\Users\PCUser\OneDrive\GitHub\garjmcmctdem_utils\scripts\conversion\active\netcdf_settings.yml"
nc_outfile = r"C:\temp\test.nc"

fnames  = []

for file in glob.glob(os.path.join(indir,"*.nc")):
    fnames.append(file)

n_files = len(fnames)

# Now parse the yaml file

settings = yaml.safe_load(open(yml_file))

# We will use the first dataset as a template to get key information about
# dimeension and variable sizes and shapes
dataset = netCDF4.Dataset(fnames[0])

# Create a dictionary for all variabes
var_dict = {}

# For each variable in our settings file get the short name, long name, units,
# dimensions and create an empty array

for key in settings["field_definitions"].keys():
    var_dict[key] = settings["field_definitions"][key].copy()
    # Lines are a special case
    if key == 'line':
        # Instead we create a line index variable. Our line varibale will be a dimension
        # with length np.uniqe(lines)
        shape = [n_files]
        dtype = np.int
        arr = np.zeros(shape = shape, dtype = dtype)
        var_dict['line_index']  = {'values': arr, 'short_name': 'line_index'}
    # For array variables
    elif key in dataset.variables.keys():
        var = dataset.variables[key]
        shape = [n_files] + list(var.shape)
        dtype = var.dtype
        arr = np.zeros(shape = shape, dtype = dtype)
        var_dict[key]['values'] = arr
    # Special flag for longitude and latitudes if they aren't in the file
    elif np.logical_or(key == 'lat',key == 'lon'):
        shape = (n_files)
        dtype = np.float64
        arr = np.zeros(shape = shape, dtype = dtype)
        var_dict[key]['values'] = arr

    # For scalar variables
    else:
        try:
            val = getattr(dataset, key)
            shape = (n_files)
            dtype = type(val)
            arr = np.zeros(shape = shape, dtype = dtype)
            var_dict[key]['values'] = arr
        except AttributeError:
            print(key, " is neither a scalar or variable. Check settings file")


# Now we populate the arrays
dataset = None
for i, file in enumerate(fnames):
    dataset = netCDF4.Dataset(file)

    for key in var_dict:
        print(key)
        # We will deal with these later
        if key == 'lat' or key == 'lon' or key == 'line':
            pass
        elif key == 'line_index':
            # We will reindex at a later time
            val = getattr(dataset, 'line')
            var_dict[key]['values'][i] = val
        elif len(var_dict[key]['values'].shape) == 1:
            # Get the scalar
            val = getattr(dataset, key)
            var_dict[key]['values'][i] = val
        else:
            # Get the variable
            arr = dataset.variables[key][:]
            var_dict[key]['values'][i] = arr
# Now we are able to create the line variable as we know which lines we are
# had data for

lines = np.unique(var_dict['line_index']['values'])
var_dict['line']['values'] = lines
# Now we want to change the line index values to index lines


for i, line in enumerate(lines):
    inds = np.where(var_dict['line_index']['values']== line)
    var_dict['line_index']['values'][inds] = i


# Now we get our longitudes and latitudes using the crs defined in the settings file
crs_projected = CRS.from_epsg(settings['crs']['projected']['epsg'])
crs_geographic = CRS.from_epsg(settings['crs']['geographic']['epsg'])

transformer = Transformer.from_crs(crs_projected, crs_geographic, always_xy=True)

lon, lat = transformer.transform(var_dict['x']['values'],var_dict['y']['values'])

var_dict['lon']['values'] = lon
var_dict['lat']['values'] = lat

# Now we want to create a dimensions dictionary with field names and sizes

dim_dict = {}

for key in settings['dimension_fields'].keys():
    dim_dict[key] = settings['dimension_fields'][key].copy()
    if key == 'line':
        dim_dict[key]['size'] = len(var_dict['line']['values'])
    else:
        dim_dict[key]['size'] = dataset.dimensions[key].size
dim_dict
# Refine the dimension definition

for key in var_dict.keys():
    # make sure all dimensions dictoinary entries are lists
    if 'dimensions' not in var_dict[key].keys():
        var_dict[key]['dimensions'] = []
    elif isinstance(var_dict[key]['dimensions'], str):
        var_dict[key]['dimensions'] = [var_dict[key]['dimensions']]
    # Extract the arrays
    arr = var_dict[key]['values']
    # If all array values are the same we can represent the variable as a scalar
    if len(np.unique(arr)) == 1:
        var_dict[key]['values'] = var_dict[key]['values'][0]
    # If we can reduce the first axis, then the variables are the same across all
    # points and we can remove this redundant axis
    elif np.all(np.unique(var_dict[key]['values'],axis = 0) == var_dict[key]['values'][0]):
        var_dict[key]['values'] = var_dict[key]['values'][0]
    # Otherwise add the point dimension to the front of the dimensions
    else:
        var_dict[key]['dimensions'].insert(0, 'point')

var_dict['line']['dimensions'] = 'line'


# Now we create a new netcdf file

rootgrp = netCDF4.Dataset(nc_outfile, "w", format="NETCDF4")

point = rootgrp.createDimension("point", None)

for key in dim_dict:
    size = dim_dict[key]['size']
    dim_name = dim_dict[key]['dimension_name']
    _ = rootgrp.createDimension(dim_name, size)

# Now add the variables

nc_vars = {}

for key in var_dict.keys():
    dat = var_dict[key]['values']
    dims = var_dict[key]['dimensions']
    dtype = dat.dtype
    short_name = var_dict[key]['short_name']
    long_name = var_dict[key]['short_name']
    if len(dims) == 0:
        nc_vars[key] = rootgrp.createVariable(short_name,dtype)
        nc_vars[key][:] = dat
        # Also make a attribute for ease of use
        rootgrp.setncattr(short_name, dat)

    else:
        nc_vars[key] = rootgrp.createVariable(short_name,dtype,dims)
        nc_vars[key][:] = dat
    nc_vars[key].long_name = long_name
    # Add units
    if 'units' in var_dict[key].keys():
        nc_vars[key].units = var_dict[key]['units']

## Add some key metadata information

rootgrp.setncattr("value_parameterization",dataset.value_parameterization)
rootgrp.setncattr("position_parameterization",dataset.position_parameterization)
rootgrp.setncattr('keywords', settings['keywords'])
rootgrp.setncattr('date_created', str(datetime.datetime.utcnow()))
rootgrp.setncattr('crs', crs_projected.name)
rootgrp.setncattr('crs_geographic',crs_geographic.name)

# Add some geospatial metdata
rootgrp.setncattr('geospatial_east_min', np.min(var_dict['x']['values']))
rootgrp.setncattr('geospatial_east_max', np.max(var_dict['x']['values']))
rootgrp.setncattr('geospatial_east_units', 'm')
rootgrp.setncattr('geospatial_north_min', np.min(var_dict['y']['values']))
rootgrp.setncattr('geospatial_north_max', np.max(var_dict['y']['values']))
rootgrp.setncattr('geospatial_north_units', 'm')
rootgrp.setncattr('geospatial_vertical_min', np.min(var_dict['elevation']['values']))
rootgrp.setncattr('geospatial_vertical_max', np.max(var_dict['elevation']['values']))
rootgrp.setncattr('geospatial_vertical_units', 'm')
rootgrp.setncattr('geospatial_lon_min', np.min(var_dict['lon']['values']))
rootgrp.setncattr('geospatial_lon_max', np.max(var_dict['lon']['values']))
rootgrp.setncattr('geospatial_lont_units', 'degrees East')
rootgrp.setncattr('geospatial_lat_min', np.min(var_dict['lat']['values']))
rootgrp.setncattr('geospatial_lon_max', np.max(var_dict['lat']['values']))
rootgrp.setncattr('geospatial_lont_units', 'degrees North')

rootgrp.close()
