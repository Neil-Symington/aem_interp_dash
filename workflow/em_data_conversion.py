'''
Created on 14/6/2020
@author: Neil Symington

This script is for converting aseg-gdf EM data to a netcdf file. The netcdf file will also include some additional
AEM system metadata.
'''

from geophys_utils.netcdf_converter import aseg_gdf2netcdf_converter
from geophys_utils.netcdf_converter.aseg_gdf_utils import aseg_gdf_format2dtype
import netCDF4
import os, math
import numpy as np
# SO we can see the logging. This enables us to debug
import gc
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")

# Define paths

root = r"C:\Users\symin\OneDrive\Documents\GA\AEM\EM"

nc_out_path = os.path.join(root, "AUS_10021_OrdK_EM.nc")

dat_in_path = os.path.join(root, 'ASEG_gdf', 'AUS_10024_InJune_EM_MGA55.dat')

dfn_in_path = os.path.join(root, 'ASEG_gdf', 'AUS_10024_InJune_EM_MGA55.dfn')

# GDA94 MGA zone 55
crs_string = "EPSG:28355"

# Initialise instance of ASEG2GDF netcdf converter
d2n = aseg_gdf2netcdf_converter.ASEGGDF2NetCDFConverter(nc_out_path,
                                                 dat_in_path,
                                                 dfn_in_path,
                                                 crs_string,
                                                 fix_precision=True,
                                                 remove_null_columns = True)
d2n.convert2netcdf()

# Now we want to parse the stm files

# Create an AEM system instance
skytem = AEM_utils.AEM_System("SkyTEM312Fast", dual_moment = True)

# Open the lm and hm files
root = r"C:\Users\PCUser\Desktop\EK_data\AEM\stm_file"

lm_file = os.path.join(root, "Skytem312Fast-LM_pV.stm")

hm_file = os.path.join(root, "Skytem312Fast-HM_pV.stm")

# Parse
skytem.parse_stm_file(lm_file, 'LM')

skytem.parse_stm_file(hm_file, 'HM')