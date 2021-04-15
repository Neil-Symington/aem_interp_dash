'''
Created on 14/6/2020
@author: Neil Symington
This script is for converting aseg-gdf conductivity data to a netcdf file.
The netcdf file will also include some additional
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


def thickness_to_depth(thickness):
    """
    Function for calculating depth top from a thickness array
    :param depth: an array of thicknesses
    :return:
    a flat array of depth
    """
    # Create a new thickness array
    depth = np.zeros(shape=thickness.shape,
                               dtype=np.float)
    # Iterate through the depth array
    depth[1:] = np.cumsum(thickness[:-1])

    return depth

# Define paths

root = "/home/nsymington/Documents/GA/AEM/Spectrem/Block1/run.05/output"

nc_out_path = os.path.join("/home/nsymington/Documents/GA/AEM/Spectrem/nc",
                           "Musgraves_block1_flight7.nc")

dat_in_path = os.path.join(root, 'inversion.output.dat')

dfn_in_path = os.path.join(root, 'inversion.output.dfn')

# Initialise instance of ASEG2GDF netcdf converter

settings_file = "/home/nsymington/PycharmProjects/garjmcmctdem_utils/utils/conversion/aseg_gdf_settings.yml"

# GDA94 MGA zone 52
crs_string = "epsg:28352"

# Initialise instance of ASEG2GDF netcdf converter

d2n = aseg_gdf2netcdf_converter.ASEGGDF2NetCDFConverter(nc_out_path,
                                                 dat_in_path,
                                                 dfn_in_path,
                                                 crs_string,
                                                 fix_precision=True,
                                                 settings_path = settings_file)

d2n.convert2netcdf()

# Create a python object with the lci dataset
d = netCDF4.Dataset(nc_out_path, "a")

layer_top_depth = np.zeros(shape = d['thickness'].shape, dtype = np.float32)

layer_top_depth = thickness_to_depth(d['thickness'][0])

layer_top_depth = np.tile(layer_top_depth, d['thickness'][:].shape[0]).reshape(d['thickness'][:].shape)

ltop = d.createVariable("layer_top_depth","f8",("point","layer"))
ltop[:] = layer_top_depth

ltop.long_name = "Depth to the top of the layer"
ltop.unit = "m"
ltop.aseg_gdf_format = "30E9.3"

d.close()