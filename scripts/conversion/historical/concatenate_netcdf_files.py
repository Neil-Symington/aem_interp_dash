import netCDF4
import os
import numpy as np
import glob

os.chdir(r"C:\Users\PCUser\Desktop\NSC_data\data\AEM\DR\garjmcmctdem_workshop\combined\pmaps")

fnames  = []

for file in glob.glob("*.nc"):
    fnames.append(file)

n_files = len(fnames)

# Numpy arrays

dataset = netCDF4.Dataset(fnames[0])
dataset


n_data = dataset.dimensions['data'].size
n_depths = dataset.dimensions['depth'].size
n_values = dataset.dimensions['value'].size
n_layers = dataset.dimensions['layer'].size
n_chains = dataset.dimensions['chain'].size
n_samples = dataset.dimensions['convergence_sample'].size

# Create a dictionary for all variabes
var_dict = {}

for key in dataset.variables.keys():
    var = dataset.variables[key]
    shape = [n_files] + list(var.shape)
    dtype = var.dtype
    arr = np.zeros(shape = shape, dtype = dtype)
    var_dict[key] = arr

# Add some more variables for scalars variables that will become arrays

vars = ['x', 'y', 'line', 'elevation', 'fiducial']
vars+= ['date', 'flight', 'nlayers_min','nlayers_max']
vars += ['nsamples', 'nchains', 'nburnin']
vars += ['thinrate', 'vmin', 'vmax', 'pmin', 'pmax']

for key in vars:
    val = getattr(dataset, key)
    shape = (n_files)
    dtype = type(val)

    if dtype == 'str':
        print(var)
        print(dtype)
    arr = np.zeros(shape = shape, dtype = dtype)
    var_dict[key] = arr


# Now we populate the arrays
dataset = None
for i, file in enumerate(glob.glob("*.nc")):
    dataset = netCDF4.Dataset(file)

    for key in var_dict:
        if len(var_dict[key].shape) == 1:
            # Get the scalar
            val = getattr(dataset, key)
            var_dict[key][i] = val
        else:
            # Get the variable
            arr = dataset.variables[key][:]
            var_dict[key][i] = arr

for key in var_dict.keys():
    arr = var_dict[key]
    if len(np.unique(arr)) == 1:
        print(key)

# Now we create a new netcdf file



rootgrp = netCDF4.Dataset(r"..\DR_rjmcmc_pmaps.nc", "w", format="NETCDF4")

point = rootgrp.createDimension("point", None)
dims = dataset.dimensions.keys()


for key in dims:
    size = dataset.dimensions[key].size
    _ = rootgrp.createDimension(key, size)

# Create a dim dict

dim_dict = {}

for key in var_dict.keys():
    try:
        dims = dataset.variables[key].dimensions
        if len(dims) == 2:
            dim_dict[key] = ("point", dims[0], dims[1])
        if len(dims) == 1:
            dim_dict[key] = ("point", dims[0])
    except KeyError:
        dim_dict[key] = ("point",)


dataset

# Now add the variables

nc_vars = {}

for key in var_dict.keys():
    arr = var_dict[key]
    if len(np.unique(arr)) == 1:
        # Make scalar to save some space
        nc_vars[key] = rootgrp.createVariable(key,arr.dtype)
        nc_vars[key] = np.unique(arr[0])
        rootgrp.setncattr(key, np.unique(arr)[0])

    else:
       dims = dim_dict[key]
       nc_vars[key] = rootgrp.createVariable(key,arr.dtype,dims)
       nc_vars[key][:] = arr

rootgrp

rootgrp.setncattr("value_parameterization",dataset.value_parameterization)
rootgrp.setncattr("position_parameterization",dataset.position_parameterization)

rootgrp.close()
