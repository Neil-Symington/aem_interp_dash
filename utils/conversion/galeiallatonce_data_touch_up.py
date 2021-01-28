# This script is used to convert the galeiallatonce dataset into a standard form that we can use with our various
# AEM interpretation and visualisation apps

import netCDF4
import numpy as np
import os
from garjmcmctdem_utils import spatial_functions

# Define paths

root = r"C:\Users\symin\OneDrive\Documents\GA\AEM\Galeiallatonce"

nc_in_path = os.path.join(root, "Mugrave_galeiallatonce.nc")


# Create a python object with the lci dataset
d = netCDF4.Dataset(nc_in_path, "a")

layer_top_depth = np.zeros(shape = d['conductivity'][:].shape, dtype = np.float32)

layer_top_depth = spatial_functions.thickness_to_depth(d['thickness'][:])

ltop = d.createVariable("layer_top_depth","f8",("point","layer"))
ltop[:] = layer_top_depth

ltop.long_name = "Depth to the top of the layer"
ltop.unit = "m"
ltop.aseg_gdf_format = "30E9.3"
d.close()