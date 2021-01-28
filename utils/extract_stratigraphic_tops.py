import pandas as pd
import netCDF4
import numpy as np
import geopandas as gpd
import sys, os
from garjmcmctdem_utils import spatial_functions, aem_utils
from shapely.geometry import Point

# Point to AEM inversion netcdf

det = aem_utils.AEM_inversion(name = "lci",
                              inversion_type = 'deterministic',
                              netcdf_dataset = netCDF4.Dataset("/home/nsymington/Documents/GA/AEM/LCI/Galilee_WB_MGA55.nc"))

infile = "/home/nsymington/Documents/GA/GAB/Galilee/Galilee_strat_tops.csv"

df_strat = pd.read_csv(infile)

long = [x for x in df_strat['Longitude'].values]
lat = [y for y in df_strat['Latitude'].values]

geometry = [Point(long[i],lat[i]) for i in range(len(long))]

gdf_strat = gpd.GeoDataFrame(df_strat, crs="EPSG:4283", geometry=geometry).to_crs("EPSG:28355")

gdf_strat['easting'] = gdf_strat['geometry'].x
gdf_strat['northing'] = gdf_strat['geometry'].y

# Remove interfaces of >400m depth

gdf_strat = gdf_strat[gdf_strat['TOP_MD_M'] < 500.]

points = np.column_stack(([val.x for val in gdf_strat['geometry']],
                          [val.y for val in gdf_strat['geometry']]))

# Remove data that are not close to the AEM lines
dist, idx = spatial_functions.nearest_neighbours(points, det.coords, points_required = 1,
                                             max_distance = 500.)

# Find the line number and fiducial of the nearest points
aem_idx = idx[idx != det.coords.shape[0]]

fids = det.data['fiducial'][aem_idx].data

lines = det.data['line'][det.data['line_index'][aem_idx]]

# remove invalid indices
bore_idx = np.where(np.isfinite(dist))

gdf_strat_ss = gdf_strat.iloc[bore_idx]
gdf_strat_ss.loc['fiducial'] = fids
gdf_strat_ss['line'] = lines

outfile = "/home/nsymington/Documents/GA/dash_data_Galilee/Galilee_strat_proximal_to_AEM.csv"

gdf_strat_ss.to_csv(outfile, index= False)

