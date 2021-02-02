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

root = r"C:\Users\symin\OneDrive\Documents\GA\AEM\Galeiallatonce"

nc_out_path = os.path.join(root, "Mugrave_galeiallatonce.nc")

dat_in_path = os.path.join(root, 'galeiallatonce.dat')

dfn_in_path = os.path.join(root, 'galeiallatonce.dfn')

# GDA94 MGA zone 52
crs_string = "epsg:28352"

# Initialise instance of ASEG2GDF netcdf converter

settings_file = "aseg_gdf_settings.yml"

d2n = aseg_gdf2netcdf_converter.ASEGGDF2NetCDFConverter(nc_out_path,
                                                 dat_in_path,
                                                 dfn_in_path,
                                                 crs_string,
                                                 fix_precision=True,
                                                 settings_path = settings_file)
d2n.convert2netcdf()
