'''
Created on 14/6/2020
@author: Neil Symington
This script is for converting aseg-gdf EM data to a netcdf file. The netcdf file will also include some additional
AEM system metadata.
'''

from geophys_utils.netcdf_converter import aseg_gdf2netcdf_converter
import netCDF4
import os, math
import numpy as np
# SO we can see the logging. This enables us to debug
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")

# Define paths

root = "/home/nsymington/Documents/GA/AEM/LCI"


nc_out_path = os.path.join(root, "Galilee_WB_MGA55.nc")

dat_in_path = os.path.join(root, 'aseg_gdf', 'Galilee_WB_MGA55.dat')

dfn_in_path = os.path.join(root, 'aseg_gdf', 'Galilee_WB_MGA55.dfn')

# GDA94 MGA zone 55
crs_string = "epsg:28355"

# Initialise instance of ASEG2GDF netcdf converter

d2n = aseg_gdf2netcdf_converter.ASEGGDF2NetCDFConverter(nc_out_path,
                                                 dat_in_path,
                                                 dfn_in_path,
                                                 crs_string,
                                                 fix_precision=True,
                                                 remove_null_columns = False)
d2n.convert2netcdf()

# Here we do some processing to ensure our lci file is somewhat standard

# Create a python object with the lci dataset
d = netCDF4.Dataset(nc_out_path, "a")

 # For consistency lets convert mS/m to S/m

d['conductivity'][:] = 0.001*d['conductivity'][:]

d['conductivity'][:].units = 'S/m'

top_layer = d['elevation'][0] - d['layer_top_elevation'][0]

top_layers = np.array([round(x,2) for x in top_layer.data])

layer_top_depth = np.zeros(shape = d['conductivity_(masked_to_DOI)'][:].shape, dtype = np.float32)

layer_top_depth[:] = np.tile(top_layers, d['conductivity'].shape[0]).reshape(d['conductivity'].shape)

ltop = d.createVariable("layer_top_depth","f8",("point","layer"))
ltop[:] = layer_top_depth

ltop.long_name = "Depth to the top of the layer"
ltop.unit = "m"
ltop.aseg_gdf_format = "30E9.3"

d.close()