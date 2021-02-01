import netCDF4
import numpy as np
import xarray as xr
from garjmcmctdem_utils.misc_utils import xarray2pickle


def nc2xarr(infile):
    d = netCDF4.Dataset(infile)
    # Define the variable name of the dimensions
    coords = {"depth": d['layer_centre_depth'][:].data,
              "conductivity_cells": d['cond_bin_centre'][:],
              "point": np.arange(d['log10conductivity_histogram'][:].shape[0])}

    dims = ['point', 'depth', 'conductivity_cells']

    # get data variables

    data_vars = {"easting": ('point', d['easting'][:].data),
                "northing": ('point', d['northing'][:].data),
                 'elevation': ('point', d['elevation'][:].data),
                'log10conductivity_histogram': (dims, d['log10conductivity_histogram'][:].data),
                "lines":    ('point', d['line'][d['line_index'][:].data].data),
                "conductivity_p10": (['point', 'layer_centre_depth'], d['conductivity_p10'][:].data),
                "conductivity_p50": (['point', 'layer_centre_depth'], d['conductivity_p50'][:].data),
                "conductivity_p90": (['point', 'layer_centre_depth'], d['conductivity_p90'][:].data),
                "interface_depth_histogram": (['point', 'layer_centre_depth'],
                                              d['interface_depth_histogram'][:].data),
                "misfit_lowest": ('point', d['misfit_lowest'][:].data),
                "fiducial": ('point', d['fiducial'][:].data),
                "layer_centre_depth": ('depth', d['layer_centre_depth'][:].data)}



    return xr.Dataset(data_vars, coords=coords)

infiles  = ["/home/nsymington/Documents/GA/dash_data_Surat/Injune_rjmcmc_pmaps.nc",
            "/home/nsymington/Documents/GA/dash_data_Surat/Injune_additional_rjmcmc_pmaps.nc",
            "/home/nsymington/Documents/GA/dash_data_Surat/Injune_rjmcmc_more_pmaps.nc"
            ]

ds = []

for item in infiles:
    ds.append(nc2xarr(item))

ds_merged = xr.concat([ds[0], ds[1]], dim = 'point')

# reprocess lines so they fit with the other implementation of producing netcdf files

lines = np.sort(np.unique(ds_merged['lines']))

# now find the index of the line
line_index = np.zeros(shape = ds_merged['lines'].shape, dtype = np.int32)


for i, line in enumerate(lines):
    inds = np.where(ds_merged['lines']== line)
    line_index[inds] = i

# drop the lines variable
ds_merged =ds_merged.drop("lines")

# add the line type variables/ coordinates to the dataset
ds_merged.coords['line'] = lines
ds_merged['line_index'] = ('point', line_index)

# finally we create a useless layer top depth

ds_merged['layer_top_depth'] = ds_merged['layer_centre_depth'][:].data

ds_merged['histogram_samples'] = ds_merged['log10conductivity_histogram'][0].data.sum(axis = 1)[0]

ds_merged.to_netcdf('/home/nsymington/Documents/GA/dash_data_Surat/Injune_pmaps_reduced_concatenated.nc')
