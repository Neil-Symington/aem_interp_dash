# This script takes a/ many netcdf

import netCDF4
import numpy as np
import xarray as xr



def nc2xarr(infile):
    d = netCDF4.Dataset(infile)
    # Here we apply a mask to all null non-convergent chains
    convergence_mask = np.where(d["misfit_lowest"][:] < 2.)[0]

    # Define the variable name of the dimensions
    coords = {"depth": d['layer_centre_depth'][:].data,
              "conductivity_cells": d['cond_bin_centre'][:],
              "point": np.arange(convergence_mask.shape[0])}

    dims = ['point', 'depth', 'conductivity_cells']

    # get data variables

    data_vars = {"easting": ('point', d['easting'][convergence_mask].data),
                "northing": ('point', d['northing'][convergence_mask].data),
                 'elevation': ('point', d['elevation'][convergence_mask].data),
                'log10conductivity_histogram': (dims, d['log10conductivity_histogram'][convergence_mask].data),
                "lines":    ('point', d['line'][d['line_index'][convergence_mask].data].data),
                "conductivity_p10": (['point', 'layer_centre_depth'], d['conductivity_p10'][convergence_mask].data),
                "conductivity_p50": (['point', 'layer_centre_depth'], d['conductivity_p50'][convergence_mask].data),
                "conductivity_p90": (['point', 'layer_centre_depth'], d['conductivity_p90'][convergence_mask].data),
                "interface_depth_histogram": (['point', 'layer_centre_depth'],
                                              d['interface_depth_histogram'][convergence_mask].data),
                "misfit_lowest": ('point', d['misfit_lowest'][convergence_mask].data),
                "fiducial": ('point', d['fiducial'][convergence_mask].data),
                "layer_centre_depth": ('depth', d['layer_centre_depth'][:].data)}



    return xr.Dataset(data_vars, coords=coords)

infiles  = ["/home/nsymington/Documents/GA/dash_data_Surat/Injune_rjmcmc_pmaps.nc",
            "/home/nsymington/Documents/GA/dash_data_Surat/Injune_additional_rjmcmc_pmaps.nc",
            "/home/nsymington/Documents/GA/dash_data_Surat/Injune_rjmcmc_more_pmaps.nc",
            "/home/nsymington/Documents/GA/dash_data_Surat/Injune_rjmcmc_even_more_pmaps.nc"
            ]

ds = []

for item in infiles:
    ds.append(nc2xarr(item))

ds_merged = xr.concat([ds[0], ds[1], ds[2], ds[3]], dim = 'point')

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

# Now we do some culling on

ds_merged.to_netcdf('/home/nsymington/Documents/GA/dash_data_Surat/Injune_pmaps_reduced_concatenated.nc')
