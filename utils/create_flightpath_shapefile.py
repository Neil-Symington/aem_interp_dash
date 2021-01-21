from garjmcmctdem_utils import aem_utils
import os
import netCDF4

infile = r"C:\Users\u77932\Documents\EFTF2\SW\data\existing\AEM\110284_Data_Package\AEM\netcdf\Mugrave_WB_MGA52.nc"

lci = aem_utils.AEM_inversion(name = 'Musgrave_LCI',
                              inversion_type = 'deterministic',
                              netcdf_dataset = netCDF4.Dataset(infile))

lci.create_flightline_polylines(crs = "EPSG:28352")

lci.flightlines.to_file(r"C:\Users\u77932\Documents\EFTF2\SW\data\existing\AEM\110284_Data_Package\AEM\vector\AEM_flightlines.shp")