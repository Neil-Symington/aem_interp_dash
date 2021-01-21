import netCDF4
import os
from garjmcmctdem_utils import aem_utils, spatial_functions, plotting_functions
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

infile = r"C:\Users\u77932\Documents\EFTF2\SW\data\existing\AEM\110284_Data_Package\AEM\netcdf\Mugrave_WB_MGA52.nc"

lci = aem_utils.AEM_inversion(name = 'Musgrave_LCI',
                              inversion_type = 'deterministic',
                              netcdf_dataset = netCDF4.Dataset(infile))

infile = r"C:\Users\u77932\Documents\EFTF2\SW\data\existing\Bores\NGIS_GFLOW3_pv.csv"

df_bores = pd.read_csv(infile)

df_bores['geometry'] = ''

# Create geodataframe
lon = df_bores['Longitude']
lat = df_bores['Latitude']


for i in range(len(lon)):
    df_bores.at[i, 'geometry'] = Point(lon[i], lat[i])

# Create a geodataframe and convert to GDA zone 52 to match the AEM dataset
gdf_bores = gpd.GeoDataFrame(df_bores, crs = "EPSG:4283").to_crs("EPSG:28352")

# Get the coordinates
gdf_bores['x'] = [coord.x for coord in gdf_bores['geometry']]
gdf_bores['y'] = [coord.y for coord in gdf_bores['geometry']]

bore_coords = gdf_bores[['x', 'y']].values

dist, inds = spatial_functions.nearest_neighbours(bore_coords, lci.coords, max_distance = 500.)

inds[np.isnan(dist)] = -9999

gdf_bores['dist'] = dist
gdf_bores['point_index'] = inds

# Now search for a specific bore

bore_id = 13761

row = gdf_bores[gdf_bores['OBJECTID'] == 13761]

# Now get the conductivity profile

cond = lci.data['conductivity'][row['point_index'].values[0] + 200]

line_ind = lci.data['line_index'][row['point_index'].values[0]]

print(lci.data['line'][line_ind])