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

root = r"C:\Users\u77932\Documents\NGS\data\AEM\easternCorridor\galei"

nc_out_path = os.path.join(r"C:\Users\u77932\Documents\workshop\AEM_Perth\data\ASEG_gdf", "AEM_workshop_galei.nc")

dat_in_path = r"C:\Users\u77932\Documents\workshop\AEM_Perth\data\ASEG_gdf\AEM_workshop_galei.dat"#os.path.join(root, 'AusAEM_East_Resources_Corridor_GA_vsum_inversion_phase1.dat')

dfn_in_path = os.path.join(root, 'AusAEM_East_Resources_Corridor_GA_vsum_inversion.dfn')


# GDA94 MGA zone 54
crs_string = "epsg:28354"

# Initialise instance of ASEG2GDF netcdf converter

settings_file = "aseg_gdf_settings.yml"

d2n = aseg_gdf2netcdf_converter.ASEGGDF2NetCDFConverter(nc_out_path,
                                                 dat_in_path,
                                                 dfn_in_path,
                                                 crs_string,
                                                 fix_precision=True,
                                                 settings_path = settings_file,
                                                 verbose = True)
d2n.convert2netcdf()

# Create a python object with the lci dataset
d = netCDF4.Dataset(nc_out_path, "a")

layer_top_depth = np.zeros(shape = d['thickness'].shape, dtype = np.float32)

layer_top_depth[:,1:] = np.cumsum(d['thickness'][:,:-1], axis = 1)

ltop = d.createVariable("layer_top_depth","f8",("point","layer"))
ltop[:] = layer_top_depth

ltop.long_name = "Depth to the top of the layer"
ltop.unit = "m"
ltop.aseg_gdf_format = "30E9.3"

d.close()